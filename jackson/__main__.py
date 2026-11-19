"""python -m jackson <command>

  fetch                       pull the pinned upstream Spoken corpus + pack builder
  stats   [selection]         how many lines/characters a selection covers
  rewrite [selection]         rewrite lines with Claude (live), or --batch / --collect ID
  import  FILE.json [--model NAME]  add rewrites made elsewhere ({lineId: text}) to the cache
  show    [selection]         print original vs rewrite side by side
  script  [selection]         write data/recording-script.md for the voice donor
  render  [selection] --voice NAME   render rewrites with F5-TTS (local GPU)
  build                       build dist/SpokenCommunity_jackson (+ .zip)
  install --addons DIR        copy the built pack into WoW's Interface/AddOns

selection: --region NAME | --quest ID... | --npc ID... | --sources KIND... | --limit N
"""
import argparse
import random

from . import config, lines as linesel, upstream


def _selection(p):
    p.add_argument("--region", choices=sorted(config.REGIONS))
    p.add_argument("--quest", type=int, nargs="+", dest="quest_ids")
    p.add_argument("--npc", type=int, nargs="+", dest="npc_ids")
    p.add_argument("--sources", nargs="+", choices=linesel.SOURCES)
    p.add_argument("--limit", type=int)


def _select(args):
    corpus = upstream.load_corpus()
    chosen = linesel.select(corpus, region=args.region, sources=args.sources,
                            quest_ids=args.quest_ids, npc_ids=args.npc_ids, limit=args.limit)
    return corpus, chosen


def _texts(chosen):
    from .rewrite import load_cache, rewritten_text
    cache = load_cache()
    return {l["lineId"]: t for l in chosen if (t := rewritten_text(cache, l))}


def main():
    ap = argparse.ArgumentParser(prog="python -m jackson", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch").add_argument("--force", action="store_true")

    _selection(sub.add_parser("stats"))

    p = sub.add_parser("rewrite")
    _selection(p)
    p.add_argument("--model", default=config.MODEL)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--force", action="store_true", help="redo lines already rewritten")
    p.add_argument("--batch", action="store_true", help="submit via the Batches API (half price)")
    p.add_argument("--collect", metavar="BATCH_ID", help="collect a submitted batch")
    p.add_argument("--wait", action="store_true", help="with --collect: poll until it ends")

    p = sub.add_parser("import")
    p.add_argument("files", nargs="+")
    p.add_argument("--model", default="claude-chat", help="recorded as the rewrite's origin")

    p = sub.add_parser("show")
    _selection(p)
    p.add_argument("--sample", type=int, default=5)

    p = sub.add_parser("script")
    _selection(p)
    p.add_argument("--count", type=int, default=60)

    p = sub.add_parser("render")
    _selection(p)
    p.add_argument("--voice", required=True, help="folder name under voices/")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--nfe-step", type=int, default=32, help="quality/speed tradeoff (16-64)")
    p.add_argument("--take", type=int, default=0, help="change to get a different reading")
    p.add_argument("--force", action="store_true", help="re-render existing clips")
    p.add_argument("--device", help="e.g. cuda, cuda:0, cpu")

    p = sub.add_parser("build")
    p.add_argument("--version", default="0.1.0")

    sub.add_parser("install").add_argument("--addons", required=True,
                                           help=r"e.g. C:\Program Files (x86)\World of Warcraft\_classic_era_\Interface\AddOns")

    args = ap.parse_args()

    if args.cmd == "fetch":
        upstream.fetch(force=args.force)

    elif args.cmd == "stats":
        _, chosen = _select(args)
        from collections import Counter
        from .rewrite import chunks
        done = _texts(chosen)
        chars = sum(len(l["text"]) for l in chosen)
        print(f"{len(chosen)} lines, {chars:,} characters, "
              f"{len({(l['npcType'], l['npcId']) for l in chosen})} speakers, "
              f"{len(chunks(chosen))} rewrite requests")
        print("by kind:", dict(Counter(l["source"] for l in chosen)))
        print(f"rewritten already: {len(done)}")

    elif args.cmd == "rewrite":
        from . import rewrite
        corpus, chosen = _select(args)
        if args.collect:
            rewrite.collect_batch(args.collect, corpus["lines"], wait=args.wait)
        elif args.batch:
            rewrite.submit_batch(chosen, model=args.model, force=args.force)
        else:
            stats = rewrite.run_live(chosen, model=args.model, concurrency=args.concurrency,
                                     force=args.force)
            print(f"done: {stats['lines']} lines rewritten, {stats['failed']} requests failed")

    elif args.cmd == "import":
        import json
        from .prompt import PROMPT_VERSION
        from .rewrite import append_cache, source_hash
        by_id = {l["lineId"]: l for l in upstream.load_corpus()["lines"]}
        records, unknown = [], 0
        for path in args.files:
            with open(path, encoding="utf-8") as f:
                for line_id, text in json.load(f).items():
                    line = by_id.get(line_id)
                    if not line or not text.strip():
                        unknown += 1
                        continue
                    records.append({"lineId": line_id, "sourceHash": source_hash(line),
                                    "promptVersion": PROMPT_VERSION, "model": args.model,
                                    "original": line["text"], "text": text.strip()})
        append_cache(records)
        print(f"imported {len(records)} rewrites ({unknown} skipped: unknown id or empty)")

    elif args.cmd == "show":
        _, chosen = _select(args)
        texts = _texts(chosen)
        have = [l for l in chosen if l["lineId"] in texts]
        for l in random.sample(have, min(args.sample, len(have))):
            title = f" — {l['questTitle']}" if l.get("questTitle") else ""
            print(f"\n=== {l['npcName']} [{l['source']}]{title}\n--- original\n{l['text']}"
                  f"\n--- jacksonified\n{texts[l['lineId']]}")

    elif args.cmd == "script":
        from . import script
        _, chosen = _select(args)
        print("wrote", script.build(chosen, _texts(chosen), count=args.count))

    elif args.cmd == "render":
        from . import render
        _, chosen = _select(args)
        print(render.render(chosen, _texts(chosen), args.voice, speed=args.speed,
                            nfe_step=args.nfe_step, take=args.take, force=args.force,
                            device=args.device))

    elif args.cmd == "build":
        from . import pack
        report = pack.build(upstream.load_corpus(), version=args.version)
        print(f"built {report['moduleDir']}: {report['audioFiles']} clips ({report['audioFormat']})")
        print(f"zip: {report['zip']}")

    elif args.cmd == "install":
        from . import pack
        print("installed to", pack.install(args.addons))


if __name__ == "__main__":
    main()
