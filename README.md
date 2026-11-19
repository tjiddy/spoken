# Spoken: Jacksonified

A private, profane community voice pack for [Spoken](https://github.com/rusty-key/spoken-wow), the
WoW Classic/Forever addon that voices quests and gossip. Every quest giver now has had it with these
motherfuckin' wolves in this motherfuckin' Northshire.

```
corpus (20k lines, from upstream Spoken)
   │  python -m jackson rewrite   ← Claude rewrites each line in the furious, foul-mouthed style
   ▼
data/rewrites.jsonl (committed)
   │  python -m jackson render    ← F5-TTS on your GPU, in a voice you're allowed to clone
   ▼
audio/{quests,gossip,followup}/*.ogg
   │  python -m jackson build     ← upstream Spoken's own pack builder, so the format matches
   ▼
dist/SpokenCommunity_jackson/  →  WoW/_classic_era_/Interface/AddOns
```

The pack declares its own voice language (`enJX`, shown as "Jacksonified"). Pick it under
**Spoken → Voice Language**; any line the pack doesn't have yet falls back to the normal English
voice, so a half-finished pack is still playable.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m jackson fetch          # pinned upstream corpus + pack builder into vendor/
```

The rewrite step needs an Anthropic API key in `ANTHROPIC_API_KEY`.

## 1. Rewrite (pilot first)

```bash
python -m jackson stats   --region ashenvale           # 382 lines, 53 speakers
python -m jackson rewrite --region ashenvale           # live requests, a few minutes
python -m jackson show    --region ashenvale --sample 5
```

Regions: `ashenvale` (both factions), `northshire`, `valley-of-trials`, `coldridge`, `deathknell`, `shadowglen`,
`camp-narache`. You can also select with `--quest ID...`, `--npc ID...`, `--sources gossip ...`,
`--limit N`.

Tweak the style in `jackson/prompt.py`; bump `PROMPT_VERSION` and rewritten lines are redone on the
next run. Lines are cached in `data/rewrites.jsonl`, so re-running only does what's missing.

**Full run** (~20k lines, Batches API at half price, usually done within the hour):

```bash
python -m jackson rewrite --batch                  # prints a batch id
python -m jackson rewrite --collect msgbatch_... --wait
python -m jackson rewrite                          # retries anything that failed, live
```

## 2. Record a voice

`python -m jackson script --region northshire` writes `data/recording-script.md` with tips and
60 lines to read. For F5-TTS zero-shot cloning you only need one 10-20 second clip:

```
voices/me/ref.wav    your best take, performed loud and angry
voices/me/ref.txt    its exact words (optional)
```

Only clone your own voice or the voice of someone who has agreed to it.

## 3. Render (on the GPU machine)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series
pip install -r requirements-tts.txt
python -m jackson render --region northshire --voice me
```

Clips that already exist are skipped, so you can stop and resume whenever you like. Useful knobs:
`--speed 1.1` (faster delivery), `--nfe-step 48` (higher quality, slower), `--take 1 --force
--quest 33` (a fresh reading of a line you didn't like).

## 4. Build and install

```bash
python -m jackson build
python -m jackson install --addons "C:\Program Files (x86)\World of Warcraft\_classic_era_\Interface\AddOns"
```

Then in game: enable the pack in the AddOns list, `/reload`, and set **Voice Language →
Jacksonified** on Spoken's settings page. To see the rewritten words in the subtitles too, also set
**Text Language** to Jacksonified, if Spoken offers it there.

## Notes

- Private use only. The pack is never published to spoken.quest or anywhere else.
- Upstream Spoken is MIT-licensed; its corpus and pack builder are fetched at a pinned commit
  (`jackson/config.py`), not vendored into this repo.
