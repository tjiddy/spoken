"""Paths and settings shared by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Upstream Spoken repo (MIT). Pinned so the corpus and the pack format don't move under us.
UPSTREAM_REPO = "https://github.com/rusty-key/spoken-wow.git"
UPSTREAM_COMMIT = "9dd73894a1a736c3cc69e17c1bb4e8111d06f928"
UPSTREAM_SPARSE = ["pipelines/quests", "apps/web/src/lib/npc"]
VENDOR_DIR = ROOT / "vendor" / "spoken-wow"
QUESTS_PIPELINE = VENDOR_DIR / "pipelines" / "quests"
CORPUS_PATH = QUESTS_PIPELINE / "corpus" / "corpus.json.gz"
IGNORED_PATH = QUESTS_PIPELINE / "corpus" / "ignored.json"

DATA_DIR = ROOT / "data"
REWRITES_PATH = DATA_DIR / "rewrites.jsonl"   # committed: the rewritten text is the valuable part
BATCHES_DIR = DATA_DIR / "batches"

AUDIO_DIR = ROOT / "audio"                     # the "store": audio/{quests,gossip,followup}/*.ogg
TAKE_TEXT_PATH = AUDIO_DIR / "take-text.json"  # words each clip says, for Spoken's subtitles
VOICES_DIR = ROOT / "voices"                   # voices/<name>/ref.wav (+ ref.txt); never committed
DIST_DIR = ROOT / "dist"

# The pack. Spoken finds a quests pack by its TOC key, not its folder name; a pack with its own
# language code shows up as its own Voice Language, and lines it lacks fall back to English.
PACK_ID = "jackson"  # becomes a Lua identifier: letters and digits only
PACK_TITLE = "Spoken Audio: Quests - Jacksonified"
PACK_LANGUAGE = "enJX"
PACK_LANGUAGE_NAME = "English (Jacksonified)"
PACK_LANGUAGE_NATIVE = "Jacksonified"

MODEL = "claude-opus-5-5"
EFFORT = "medium"
CHUNK_SIZE = 12  # lines per request; one request never mixes speakers

# Pilot regions: (map id, x range, y range) in world coordinates, matched against NPC spawns.
REGIONS = {
    "northshire": (0, (-9150.0, -8650.0), (-450.0, 150.0)),
    "valley-of-trials": (1, (-750.0, -150.0), (-4550.0, -3850.0)),
    "coldridge": (0, (-6400.0, -5900.0), (150.0, 750.0)),
    "deathknell": (0, (1650.0, 2050.0), (1350.0, 1850.0)),
    "shadowglen": (1, (10100.0, 10700.0), (600.0, 1150.0)),
    "camp-narache": (1, (-3200.0, -2700.0), (-450.0, 100.0)),
}
