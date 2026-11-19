"""A recording script for whoever lends their voice, built from real rewritten lines."""
import random

from . import config

GUIDE = """\
# Recording script

Thanks for lending your voice. A few tips make a big difference:

- **Room:** quiet, soft (a closet full of clothes is great), no fans or AC humming.
- **Mic:** same mic, same distance (about a hand's width), same gain for every take.
- **Delivery:** perform it. The clone copies the *energy* of these recordings, so if you read
  calmly you'll get a calm clone. Be loud, be outraged, be menacing, swear like you mean it.
- **Format:** WAV, mono, 24 kHz or higher. Leave half a second of silence at each end.

## For zero-shot cloning (F5-TTS)

You only need **one** clip: pick your best 10–20 second take of a single line below, save it as
`voices/<name>/ref.wav`, and put its exact words in `voices/<name>/ref.txt`. Short and clean beats
long and noisy. Try two or three different reference clips; each one gives a different flavour.

## For fine-tuning later (optional)

Record all of the lines below, one file per line, named by its number (`001.wav`, `002.wav`, ...).
That's roughly 15–25 minutes of audio.

---
"""


def build(lines: list, texts: dict, count: int = 60, seed: int = 7) -> str:
    """Pick a varied set: every kind of line, short and long, many speakers."""
    pool = [(l, texts[l["lineId"]]) for l in lines if texts.get(l["lineId"])]
    if not pool:
        raise SystemExit("no rewritten lines yet; run `rewrite` on a pilot first")
    rng = random.Random(seed)
    rng.shuffle(pool)
    by_kind = {}
    for item in pool:
        by_kind.setdefault(item[0]["source"], []).append(item)
    picked, seen_speakers = [], set()
    # Round-robin across kinds, preferring new speakers and moderate lengths (5-60 s spoken).
    while len(picked) < count and any(by_kind.values()):
        for kind in list(by_kind):
            items = by_kind[kind]
            if not items:
                continue
            best = next((i for i in items if i[0]["npcName"] not in seen_speakers
                         and 60 <= len(i[1]) <= 700), items[0])
            items.remove(best)
            picked.append(best)
            seen_speakers.add(best[0]["npcName"])
            if len(picked) >= count:
                break
    out = [GUIDE]
    for n, (line, text) in enumerate(picked, 1):
        title = f" — {line['questTitle']}" if line.get("questTitle") else ""
        out.append(f"### {n:03d}  ({line['source']}, {line['npcName']}{title})\n\n{text}\n")
    path = config.DATA_DIR / "recording-script.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)
