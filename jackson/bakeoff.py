"""Render the same handful of lines through each engine and compare them by ear.

Every run adds one column to bakeoff/index.html, so engines that need different Python
environments (or a different model loaded in the server) can be run one after another.
"""
import html
import json
import time

from . import config

# A spread of what the pack has to do: long briefings, a menacing punchline, a gentle one, a
# short check-in and a one-liner of directions. All from the Ashenvale pilot.
SAMPLE_LINES = [
    "q:1054:accept",                          # Raene Wolfrunner: the furbolg briefing, long
    "q:6564:complete",                        # Je'neu Sancrea: "Lorgus. Motherfuckin'. Jett."
    "q:6621:complete",                        # Karang Amakkar: orc commander, proud
    "q:1035:accept",                          # Pelturas Whitemoon: desperate father, quieter
    "q:1017:complete",                        # Sentinel Velene Starstrike: gritty reward
    "g:927f2bfc50de1f03c38ac6ec3198f4a7",     # an innkeeper: "Lots of goddamn water."
]

BAKEOFF_DIR = config.ROOT / "bakeoff"


def run(corpus: dict, texts: dict, voice: str, label: str, engine_kwargs: dict,
        line_ids=None) -> dict:
    import soundfile as sf
    from .render import make_engine, tts_text, render_line

    by_id = {l["lineId"]: l for l in corpus["lines"]}
    line_ids = line_ids or SAMPLE_LINES
    missing = [i for i in line_ids if i not in texts]
    if missing:
        raise SystemExit(f"no rewrite yet for {missing}; run the Ashenvale rewrite first")

    out_dir = BAKEOFF_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_dir = config.VOICES_DIR / voice
    print(f"loading {label} ...")
    t0 = time.time()
    tts = make_engine(voice_dir=voice_dir, **engine_kwargs)
    load_s = time.time() - t0

    results = []
    for n, line_id in enumerate(line_ids, 1):
        t = time.time()
        audio, sr, _ = render_line(tts, tts_text(texts[line_id]), line_id)
        took = time.time() - t
        sf.write(str(out_dir / f"{n:02d}.wav"), audio, sr)
        secs = len(audio) / sr
        results.append({"n": n, "lineId": line_id, "seconds": round(secs, 1),
                        "render_s": round(took, 1)})
        print(f"  {n:02d} {by_id[line_id]['npcName']}: {secs:.1f}s of audio in {took:.1f}s")

    meta = {"label": label, "engine": {k: v for k, v in engine_kwargs.items() if k != "server"},
            "server": {k: v for k, v in (engine_kwargs.get("server") or {}).items()},
            "load_s": round(load_s, 1), "lines": results, "voice": voice,
            "when": time.strftime("%Y-%m-%d %H:%M")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    write_index(corpus, texts)
    return meta


def write_index(corpus: dict, texts: dict) -> str:
    """bakeoff/index.html: one row per line, one column per engine, plus the reference clip."""
    by_id = {l["lineId"]: l for l in corpus["lines"]}
    runs = []
    for d in sorted(BAKEOFF_DIR.iterdir()):
        if (d / "meta.json").exists():
            runs.append(json.loads((d / "meta.json").read_text(encoding="utf-8")))
    if not runs:
        return None
    line_ids = [r["lineId"] for r in runs[0]["lines"]]
    voice = runs[-1]["voice"]
    head = "".join(
        f"<th>{html.escape(r['label'])}<small>{html.escape(r['when'])}"
        f"{' · ' + html.escape(str(r['server'].get('model'))) if r['server'] else ''}</small></th>"
        for r in runs)
    rows = []
    for n, line_id in enumerate(line_ids, 1):
        line = by_id.get(line_id, {})
        cells = []
        for r in runs:
            item = next((x for x in r["lines"] if x["lineId"] == line_id), None)
            cells.append(
                f'<td><audio controls preload="none" src="{html.escape(r["label"])}/{n:02d}.wav"></audio>'
                f'<small>{item["seconds"]}s audio, {item["render_s"]}s to render</small></td>'
                if item else "<td>—</td>")
        rows.append(
            f"<tr><td class=line><b>{html.escape(line.get('npcName', line_id))}</b> "
            f"<small>{html.escape(line.get('source', ''))}</small>"
            f"<p>{html.escape(texts.get(line_id, ''))}</p></td>{''.join(cells)}</tr>")
    page = f"""<!doctype html><meta charset=utf-8><title>Voice Bakeoff</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e3e3e8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151517;--fg:#ececf0;--muted:#9a9aa3;--line:#2c2c31}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:24px}}
table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid var(--line);padding:10px;vertical-align:top;text-align:left}}
th small,td small{{display:block;color:var(--muted);font-weight:400;font-size:12px}}
td.line{{max-width:420px}} td.line p{{margin:.4em 0 0;color:var(--muted);white-space:pre-wrap}}
audio{{width:220px}}
</style>
<h1>Voice bakeoff</h1>
<p>Reference clip ({html.escape(voice)}): <audio controls src="../voices/{html.escape(voice)}/ref.wav"></audio></p>
<table><tr><th>Line</th>{head}</tr>{''.join(rows)}</table>"""
    path = BAKEOFF_DIR / "index.html"
    path.write_text(page, encoding="utf-8")
    return str(path)
