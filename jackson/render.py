"""Render rewritten lines to audio with F5-TTS on a local GPU.

Voice folder layout (never committed):
    voices/<name>/ref.wav   10-20 s of clean speech in the voice to clone, in the delivery you want
    voices/<name>/ref.txt   exactly what is said in ref.wav (optional: F5 transcribes it if absent)

Output goes into the store, audio/<subfolder>/<fileName>.ogg, plus audio/take-text.json so the
pack's subtitles show the words actually spoken. Already-rendered lines are skipped, so a run
can be stopped and resumed.
"""
import hashlib
import json
import re

from . import config
from .lines import stem

_STAGE = re.compile(r"<[^<>]*>")


def tts_text(text: str) -> str:
    """Last-chance cleanup of anything a TTS engine would read literally."""
    text = _STAGE.sub(" ", text)
    text = text.replace("*", "").replace("_", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def _seed(line_id: str, take: int) -> int:
    """Stable per line, so re-rendering a line reproduces it unless you ask for a new take."""
    return int(hashlib.sha1(f"{line_id}:{take}".encode()).hexdigest()[:8], 16)


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


def render(lines: list, texts: dict, voice: str, speed: float = 1.0, nfe_step: int = 32,
           take: int = 0, force: bool = False, device: str = None) -> dict:
    """`texts` maps lineId to the words to speak (the rewrite)."""
    import soundfile as sf
    from f5_tts.api import F5TTS
    from tqdm import tqdm

    voice_dir = config.VOICES_DIR / voice
    ref_wav = voice_dir / "ref.wav"
    if not ref_wav.exists():
        raise SystemExit(f"no reference clip at {ref_wav}")
    ref_txt_path = voice_dir / "ref.txt"
    ref_text = ref_txt_path.read_text(encoding="utf-8").strip() if ref_txt_path.exists() else ""

    todo = []
    for line in lines:
        words = texts.get(line["lineId"])
        if not words:
            continue
        out = config.AUDIO_DIR / f"{stem(line)}.ogg"
        if out.exists() and not force:
            continue
        todo.append((line, tts_text(words), out))
    print(f"{len(todo)} lines to render with voice '{voice}'")
    if not todo:
        return {"rendered": 0}

    tts = F5TTS(device=device) if device else F5TTS()
    if not ref_text:
        # Transcribe the reference once (Whisper) and save it, rather than on every line.
        from f5_tts.infer.utils_infer import preprocess_ref_audio_text
        _, ref_text = preprocess_ref_audio_text(str(ref_wav), "", show_info=print)
        ref_txt_path.write_text(ref_text.strip() + "\n", encoding="utf-8")
        print(f"transcribed reference: {ref_text.strip()!r} (saved to {ref_txt_path}; fix it if wrong)")
    take_text = load_take_text()
    done = 0
    try:
        for line, words, out in tqdm(todo, unit="line"):
            out.parent.mkdir(parents=True, exist_ok=True)
            wav, sr, _ = tts.infer(
                ref_file=str(ref_wav), ref_text=ref_text, gen_text=words,
                speed=speed, nfe_step=nfe_step, seed=_seed(line["lineId"], take),
                show_info=lambda *a, **k: None, progress=None,
            )
            # Ogg Vorbis straight from libsndfile: what the official packs ship, no ffmpeg needed.
            sf.write(str(out), wav, sr, format="OGG", subtype="VORBIS")
            take_text[stem(line)] = words
            done += 1
            if done % 25 == 0:
                save_take_text(take_text)
    finally:
        save_take_text(take_text)
    return {"rendered": done}
