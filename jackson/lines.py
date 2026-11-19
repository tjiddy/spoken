"""Choosing which corpus lines a run works on."""
from . import config
from .upstream import tts_cli

SOURCES = ("accept", "progress", "complete", "gossip", "followup")


def stem(line: dict) -> str:
    """Store-relative stem the addon resolves the line by, e.g. 'quests/33-complete'."""
    return f"{tts_cli().naming.subfolder_from_line_id(line['lineId'])}/{line['fileName']}"


def _spawn_index(corpus: dict) -> dict:
    spawns = corpus.get("spawns") or {}
    return dict(spawns) if isinstance(spawns, list) else spawns


def in_region(line: dict, spawns: dict, region: str) -> bool:
    """True when most of the speaker's spawns are inside the region's box, so an NPC who stands
    in every city (the Brigadier General) doesn't count for each zone he happens to visit."""
    map_id, (x0, x1), (y0, y1) = config.REGIONS[region]
    spots = spawns.get(f"{line['npcType']}:{line['npcId']}") or ()
    inside = sum(1 for s in spots
                 if s.get("map") == map_id and x0 <= s["x"] <= x1 and y0 <= s["y"] <= y1)
    return inside * 2 > len(spots)


def select(corpus: dict, region: str = None, sources=None, quest_ids=None, npc_ids=None,
           limit: int = None) -> list:
    """Generatable lines, filtered. Region matches on where the speaker spawns."""
    spawns = _spawn_index(corpus) if region else None
    sources = set(sources or SOURCES)
    chosen = []
    for line in corpus["lines"]:
        if not line.get("generatable") or not (line.get("text") or "").strip():
            continue
        if line["source"] not in sources:
            continue
        if quest_ids and line.get("questId") not in quest_ids:
            continue
        if npc_ids and line.get("npcId") not in npc_ids:
            continue
        if region and not in_region(line, spawns, region):
            continue
        chosen.append(line)
    return chosen[:limit] if limit else chosen
