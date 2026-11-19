"""Fetch the pinned upstream Spoken repo and expose its corpus and pack builder."""
import subprocess
import sys

from . import config


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch(force: bool = False) -> None:
    """Sparse, shallow checkout of just the quests pipeline at the pinned commit."""
    vendor = config.VENDOR_DIR
    if vendor.exists() and not force:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=vendor, capture_output=True,
                              text=True).stdout.strip()
        if head == config.UPSTREAM_COMMIT:
            print(f"upstream already at {head[:10]} in {vendor}")
            return
    vendor.mkdir(parents=True, exist_ok=True)
    if not (vendor / ".git").exists():
        _git("init", "-q", cwd=vendor)
        _git("remote", "add", "origin", config.UPSTREAM_REPO, cwd=vendor)
    _git("sparse-checkout", "set", *config.UPSTREAM_SPARSE, cwd=vendor)
    _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", config.UPSTREAM_COMMIT,
         cwd=vendor)
    _git("checkout", "-q", "FETCH_HEAD", cwd=vendor)
    print(f"upstream checked out at {config.UPSTREAM_COMMIT[:10]} in {vendor}")


def _require():
    if not config.CORPUS_PATH.exists():
        sys.exit("upstream not fetched yet: run `python -m jackson fetch` first")
    path = str(config.QUESTS_PIPELINE)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_corpus() -> dict:
    _require()
    from tts_cli.corpus import load_corpus as upstream_load
    return upstream_load(str(config.CORPUS_PATH))


def tts_cli():
    """The upstream package, importable once fetched."""
    _require()
    import tts_cli  # noqa: F401
    import tts_cli.build
    import tts_cli.ignores
    import tts_cli.naming
    return tts_cli
