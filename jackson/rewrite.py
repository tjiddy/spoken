"""Rewrite corpus lines with Claude, cached in data/rewrites.jsonl.

Two modes:
- live (default): concurrent requests, good for pilots of a few hundred lines.
- batch: submit to the Message Batches API (half price, usually done within the hour), then
  `collect` the results later. Meant for the full ~20k-line run.
"""
import asyncio
import hashlib
import json
import time
from collections import defaultdict

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from . import config
from .prompt import OUTPUT_SCHEMA, PROMPT_VERSION, SYSTEM, user_message

# Server-side refusal fallback (live mode only; the Batches API does not accept it).
FALLBACK_BETA = "server-side-fallback-2026-07-01"


# ---------------------------------------------------------------- cache

def source_hash(line: dict) -> str:
    return hashlib.sha1(line["text"].encode("utf-8")).hexdigest()[:12]


def load_cache() -> dict:
    """{lineId: record}, last write wins."""
    cache = {}
    if config.REWRITES_PATH.exists():
        with open(config.REWRITES_PATH, encoding="utf-8") as f:
            for raw in f:
                if raw.strip():
                    rec = json.loads(raw)
                    cache[rec["lineId"]] = rec
    return cache


def is_fresh(rec: dict, line: dict) -> bool:
    return bool(rec) and rec.get("sourceHash") == source_hash(line) \
        and rec.get("promptVersion") == PROMPT_VERSION


def append_cache(records: list) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REWRITES_PATH, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def rewritten_text(cache: dict, line: dict):
    rec = cache.get(line["lineId"])
    return rec["text"] if is_fresh(rec, line) else None


# ---------------------------------------------------------------- requests

def pending(lines: list, cache: dict, force: bool = False) -> list:
    return [l for l in lines if force or not is_fresh(cache.get(l["lineId"]), l)]


def chunks(lines: list) -> list:
    """Group by speaker (consistent character), then split into CHUNK_SIZE pieces."""
    by_speaker = defaultdict(list)
    for line in lines:
        by_speaker[(line["npcType"], line["npcId"], line["npcName"])].append(line)
    out = []
    for group in by_speaker.values():
        for i in range(0, len(group), config.CHUNK_SIZE):
            out.append(group[i:i + config.CHUNK_SIZE])
    return out


def request_params(chunk: list, model: str) -> dict:
    return dict(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_message(chunk[0], chunk)}],
        output_config={"effort": config.EFFORT,
                       "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )


def parse(message, chunk: list, model: str) -> list:
    """Records for a chunk, or raises if the response is unusable."""
    if message.stop_reason == "refusal":
        raise RuntimeError(f"refused: {getattr(message, 'stop_details', None)}")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("hit max_tokens")
    text = next((b.text for b in message.content if b.type == "text"), None)
    if not text:
        raise RuntimeError("no text block in response")
    by_id = {item["id"]: item["text"].strip() for item in json.loads(text)["lines"]}
    records = []
    for i, line in enumerate(chunk):
        new = by_id.get(str(i))
        if not new:
            raise RuntimeError(f"missing id {i} ({line['lineId']})")
        records.append({
            "lineId": line["lineId"],
            "sourceHash": source_hash(line),
            "promptVersion": PROMPT_VERSION,
            "model": model,
            "original": line["text"],
            "text": new,
        })
    return records


# ---------------------------------------------------------------- live mode

async def _run_live(todo: list, model: str, concurrency: int) -> dict:
    client = anthropic.AsyncAnthropic(max_retries=5)
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "failed": 0, "lines": 0}

    async def one(chunk):
        async with sem:
            try:
                msg = await client.beta.messages.create(
                    **request_params(chunk, model),
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                )
                records = parse(msg, chunk, model)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError,
                    json.JSONDecodeError, KeyError) as e:
                stats["failed"] += 1
                print(f"  ! {chunk[0]['npcName']} ({len(chunk)} lines): {e}")
                return
            append_cache(records)  # written as they land, so an interrupted run keeps its work
            stats["ok"] += 1
            stats["lines"] += len(records)
            print(f"  ✓ {chunk[0]['npcName']}: {len(records)} lines")

    await asyncio.gather(*(one(c) for c in chunks(todo)))
    return stats


def run_live(lines: list, model: str = config.MODEL, concurrency: int = 6,
             force: bool = False) -> dict:
    todo = pending(lines, load_cache(), force)
    print(f"{len(todo)} of {len(lines)} lines need rewriting "
          f"({len(chunks(todo))} requests to {model})")
    if not todo:
        return {"ok": 0, "failed": 0, "lines": 0}
    return asyncio.run(_run_live(todo, model, concurrency))


# ---------------------------------------------------------------- batch mode

def submit_batch(lines: list, model: str = config.MODEL, force: bool = False) -> str:
    todo = pending(lines, load_cache(), force)
    groups = chunks(todo)
    if not groups:
        print("nothing to rewrite")
        return None
    manifest = {f"c{i:05d}": [l["lineId"] for l in g] for i, g in enumerate(groups)}
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=[
        Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**request_params(g, model)))
        for cid, g in zip(manifest, groups)
    ])
    config.BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BATCHES_DIR / f"{batch.id}.json", "w", encoding="utf-8") as f:
        json.dump({"model": model, "chunks": manifest}, f)
    print(f"submitted batch {batch.id}: {len(todo)} lines in {len(groups)} requests")
    return batch.id


def collect_batch(batch_id: str, corpus_lines: list, wait: bool = False) -> dict:
    client = anthropic.Anthropic()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended" or not wait:
            break
        c = batch.request_counts
        print(f"  {batch.processing_status}: {c.processing} processing, {c.succeeded} done")
        time.sleep(60)
    if batch.processing_status != "ended":
        print(f"batch {batch_id} is still {batch.processing_status}; try again later or pass --wait")
        return None
    with open(config.BATCHES_DIR / f"{batch_id}.json", encoding="utf-8") as f:
        manifest = json.load(f)
    by_id = {l["lineId"]: l for l in corpus_lines}
    stats = {"ok": 0, "failed": 0, "lines": 0}
    for result in client.messages.batches.results(batch_id):
        chunk = [by_id[i] for i in manifest["chunks"][result.custom_id] if i in by_id]
        if result.result.type != "succeeded":
            stats["failed"] += 1
            print(f"  ! {result.custom_id}: {result.result.type}")
            continue
        try:
            records = parse(result.result.message, chunk, manifest["model"])
        except (RuntimeError, json.JSONDecodeError, KeyError) as e:
            stats["failed"] += 1
            print(f"  ! {result.custom_id}: {e}")
            continue
        append_cache(records)
        stats["ok"] += 1
        stats["lines"] += len(records)
    print(f"collected {stats['lines']} lines; {stats['failed']} requests failed "
          "(re-run `rewrite` to retry just those)")
    return stats
