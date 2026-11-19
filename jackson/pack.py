"""Build the Spoken community pack with upstream's own builder, so the format always matches."""
import os
import shutil
import zipfile

from . import config
from .upstream import tts_cli


def module_name() -> str:
    return f"SpokenCommunity_{config.PACK_ID}"


def build(corpus: dict, version: str = "0.1.0") -> dict:
    cli = tts_cli()
    if not any(config.AUDIO_DIR.glob("*/*.ogg")):
        raise SystemExit(f"no audio in {config.AUDIO_DIR}; run `render` first")

    # Upstream's builder wants a per-language icon; ours reuses the English one.
    icon_dir = config.QUESTS_PIPELINE / "assets" / "icon"
    lang_icon = icon_dir / f"spoken-quests-{config.PACK_LANGUAGE}.tga"
    if not lang_icon.exists():
        shutil.copy2(icon_dir / "spoken-quests.tga", lang_icon)

    ignored = cli.ignores.load_pack_ignored(corpus, str(config.IGNORED_PATH))
    report = cli.build.build_module(
        corpus, str(config.AUDIO_DIR), dist_dir=str(config.DIST_DIR),
        module_name=module_name(), version=version, title=config.PACK_TITLE,
        language=config.PACK_LANGUAGE, ignored=ignored,
    )

    # Language names, so Spoken's Voice Language menu can show ours (community/README.md).
    toc = config.DIST_DIR / module_name() / f"{module_name()}.toc"
    extra = (f"## X-Spoken-LanguageName: {config.PACK_LANGUAGE_NAME}\n"
             f"## X-Spoken-LanguageNative: {config.PACK_LANGUAGE_NATIVE}\n")
    toc.write_text(extra + toc.read_text(encoding="utf-8"), encoding="utf-8")

    report["zip"] = str(_zip(config.DIST_DIR / module_name()))
    return report


def _zip(module_dir):
    """One folder at the zip root, forward slashes: what the Spoken Manager accepts."""
    path = module_dir.with_suffix(".zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(module_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(module_dir.parent).as_posix())
    return path


def install(addons_dir: str) -> str:
    src = config.DIST_DIR / module_name()
    if not src.exists():
        raise SystemExit("nothing built yet; run `build` first")
    target = os.path.join(addons_dir, module_name())
    if os.path.exists(target):
        shutil.rmtree(target)
    shutil.copytree(src, target)
    return target
