"""Render rewritten lines to audio on a local GPU.

Engines:
  chatterbox (default)  Resemble AI's Chatterbox: the most natural-sounding of the two, with an
                        `exaggeration` knob for intensity. Best for the furious delivery.
  f5                    F5-TTS: lighter and faster, decent zero-shot clone.

Quality control ("best of N"): every line is split into sentence-sized chunks; each chunk is
rendered up to --takes times with different seeds, Whisper transcribes each take, and the take
whose words match the script best wins. A take that gets every word right ends the search early.
This catches the skipped words, repeats and mumbles every cloning model produces now and then.
The finished line is loudness-normalised so every NPC plays at the same volume.

Voice folder layout (never committed):
    voices/<name>/ref.wav   10-20 s of clean speech in the voice to clone, delivered how you want it
    voices/<name>/ref.txt   exactly what is said in ref.wav (F5 only; transcribed if absent)
"""
import hashlib
import json
import re
import time

from . import config
from .lines import stem

_STAGE = re.compile(r"<[^<>]*>")
TARGET_LUFS = -16.0
CHUNK_CHARS = 280            # Chatterbox makes at most ~40 s per call; this keeps a chunk well under
SENTENCE_GAP = 0.16          # seconds of silence between chunks
PARAGRAPH_GAP = 0.45


def tts_text(text: str) -> str:
    """Last-chance cleanup of anything a TTS engine would read literally."""
    text = _STAGE.sub(" ", text)
    text = text.replace("*", "").replace("_", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def chunk_text(text: str, limit: int = CHUNK_CHARS) -> list:
    """[(chunk, gap_after_seconds)]: sentences packed up to `limit`, never across a paragraph."""
    out = []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    for pi, para in enumerate(paragraphs):
        sentences = re.findall(r"[^.!?]+(?:[.!?]+[\"')\]]*|$)", para)
        sentences = [s.strip() for s in sentences if s.strip()]
        current = ""
        for s in sentences:
            if current and len(current) + 1 + len(s) > limit:
                out.append([current, SENTENCE_GAP])
                current = s
            else:
                current = f"{current} {s}".strip()
        if current:
            out.append([current, SENTENCE_GAP])
        if out and pi < len(paragraphs) - 1:
            out[-1][1] = PARAGRAPH_GAP
    if out:
        out[-1][1] = 0.0
    return [tuple(c) for c in out]


def _seed(line_id: str, chunk: int, take: int) -> int:
    """Stable, so re-rendering reproduces a line unless you ask for --variant."""
    return int(hashlib.sha1(f"{line_id}:{chunk}:{take}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- engines

class Chatterbox:
    def __init__(self, voice_dir, device, exaggeration, cfg_weight, temperature):
        import torch
        from chatterbox.tts import ChatterboxTTS
        self.torch = torch
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.model.prepare_conditionals(str(voice_dir / "ref.wav"), exaggeration=exaggeration)
        self.sr = self.model.sr
        self.kw = dict(exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)

    def say(self, text, seed):
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        wav = self.model.generate(text, **self.kw)
        return wav.squeeze(0).cpu().numpy()


class F5:
    def __init__(self, voice_dir, device, speed, nfe_step):
        from f5_tts.api import F5TTS
        self.model = F5TTS(device=device) if device else F5TTS()
        self.ref_wav = str(voice_dir / "ref.wav")
        ref_txt = voice_dir / "ref.txt"
        self.ref_text = ref_txt.read_text(encoding="utf-8").strip() if ref_txt.exists() else ""
        if not self.ref_text:
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text
            _, self.ref_text = preprocess_ref_audio_text(self.ref_wav, "", show_info=print)
            ref_txt.write_text(self.ref_text.strip() + "\n", encoding="utf-8")
            print(f"transcribed reference: {self.ref_text.strip()!r} (saved to {ref_txt})")
        self.sr = self.model.target_sample_rate
        self.kw = dict(speed=speed, nfe_step=nfe_step)

    def say(self, text, seed):
        wav, _, _ = self.model.infer(ref_file=self.ref_wav, ref_text=self.ref_text, gen_text=text,
                                     seed=seed, show_info=lambda *a, **k: None, progress=None,
                                     **self.kw)
        return wav


# ---------------------------------------------------------------- verification

_NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
            "7": "seven", "8": "eight", "9": "nine", "10": "ten", "12": "twelve", "15": "fifteen",
            "20": "twenty", "30": "thirty", "50": "fifty", "100": "hundred"}


def _words(text: str) -> list:
    """Comparable words: 'motherfuckin'' and 'motherfucking' match, as do 'you've' and 'youve'."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"in'(?=\W|$)", "ing", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text.replace("'", ""))
    return [_NUMBERS.get(w, w) for w in text.split()]


def word_error_rate(expected: str, heard: str) -> float:
    a, b = _words(expected), _words(heard)
    if not a:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, wb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb))
        prev = cur
    return prev[-1] / len(a)


class Verifier:
    """Whisper through transformers, which Chatterbox already installs."""

    def __init__(self, device, model="openai/whisper-large-v3-turbo"):
        import torch
        from transformers import pipeline
        self.asr = pipeline("automatic-speech-recognition", model=model,
                            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                            device=device)

    def hear(self, wav, sr) -> str:
        return self.asr({"raw": wav, "sampling_rate": sr},
                        generate_kwargs={"language": "english"})["text"]


# ---------------------------------------------------------------- the loop

def _normalise(wav, sr):
    import numpy as np
    import pyloudnorm
    loudness = pyloudnorm.Meter(sr).integrated_loudness(wav)
    if np.isfinite(loudness):
        wav = pyloudnorm.normalize.loudness(wav, loudness, TARGET_LUFS)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    return wav * (0.97 / peak) if peak > 0.97 else wav


def load_take_text() -> dict:
    if config.TAKE_TEXT_PATH.exists():
        with open(config.TAKE_TEXT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_take_text(table: dict) -> None:
    config.TAKE_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.TAKE_TEXT_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=0, sort_keys=True)
    tmp.replace(config.TAKE_TEXT_PATH)


def render(lines: list, texts: dict, voice: str, engine: str = "chatterbox", takes: int = 3,
           verify: bool = True, max_wer: float = 0.0, variant: int = 0, force: bool = False,
           device: str = "cuda", exaggeration: float = 0.7, cfg_weight: float = 0.35,
           temperature: float = 0.8, speed: float = 1.0, nfe_step: int = 32) -> dict:
    """`texts` maps lineId to the words to speak (the rewrite)."""
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm

    voice_dir = config.VOICES_DIR / voice
    if not (voice_dir / "ref.wav").exists():
        raise SystemExit(f"no reference clip at {voice_dir / 'ref.wav'}")

    todo = []
    for line in lines:
        words = texts.get(line["lineId"])
        out = config.AUDIO_DIR / f"{stem(line)}.ogg"
        if words and (force or not out.exists()):
            todo.append((line, tts_text(words), out))
    print(f"{len(todo)} lines to render with {engine}, voice '{voice}', "
          f"best of {takes}{' (Whisper-checked)' if verify and takes > 1 else ''}")
    if not todo:
        return {"rendered": 0}

    if engine == "chatterbox":
        tts = Chatterbox(voice_dir, device, exaggeration, cfg_weight, temperature)
    elif engine == "f5":
        tts = F5(voice_dir, device, speed, nfe_step)
    else:
        raise SystemExit(f"unknown engine {engine!r}")
    checker = Verifier(device) if verify and takes > 1 else None

    take_text = load_take_text()
    report_path = config.AUDIO_DIR / "render-report.jsonl"
    stats = {"rendered": 0, "chunks": 0, "retakes": 0, "imperfect": 0}
    started = time.time()
    try:
        for line, words, out in tqdm(todo, unit="line"):
            pieces, worst = [], 0.0
            for ci, (chunk, gap) in enumerate(chunk_text(words)):
                best = None
                for t in range(takes):
                    wav = tts.say(chunk, _seed(line["lineId"], ci, variant * 100 + t))
                    if not checker:
                        best = (0.0, wav, None)
                        break
                    heard = checker.hear(wav, tts.sr)
                    wer = word_error_rate(chunk, heard)
                    if best is None or wer < best[0]:
                        best = (wer, wav, heard)
                    if wer <= max_wer:
                        break
                    stats["retakes"] += 1
                wer, wav, heard = best
                worst = max(worst, wer)
                stats["chunks"] += 1
                pieces.append(wav)
                if gap:
                    pieces.append(np.zeros(int(gap * tts.sr), dtype=wav.dtype))
                if checker and wer > max_wer:
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"lineId": line["lineId"], "chunk": ci, "wer": round(wer, 3),
                                            "expected": chunk, "heard": heard}) + "\n")
            audio = _normalise(np.concatenate(pieces).astype("float32"), tts.sr)
            out.parent.mkdir(parents=True, exist_ok=True)
            # Ogg Vorbis straight from libsndfile: what the official packs ship.
            sf.write(str(out), audio, tts.sr, format="OGG", subtype="VORBIS")
            take_text[stem(line)] = words
            stats["rendered"] += 1
            stats["imperfect"] += worst > max_wer
            if stats["rendered"] % 20 == 0:
                save_take_text(take_text)
    finally:
        save_take_text(take_text)
    stats["minutes"] = round((time.time() - started) / 60, 1)
    if stats["imperfect"]:
        print(f"{stats['imperfect']} lines kept a take Whisper didn't hear perfectly; see "
              f"{report_path} (often harmless: names and swear words trip Whisper too)")
    return stats
