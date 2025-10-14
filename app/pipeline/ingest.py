# app/pipeline/ingest.py
from pathlib import Path
import subprocess
import shutil
import re
import yt_dlp

AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def _slugify(text: str, max_len: int = 100) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "untitled"

def _unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    i = 1
    while True:
        p = base.with_name(f"{stem}-{i}{suffix}")
        if not p.exists():
            return p
        i += 1

def download_youtube(url: str) -> tuple[str, Path]:
    """Download YouTube audio and name it using uploader + title."""
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title") or info.get("id") or "untitled"
    uploader = info.get("uploader") or "unknown_creator"

    # Combine creator + title
    combined = f"{title}_by_{uploader}"
    slug = _slugify(combined, max_len=120)

    out_mp3 = _unique_path(AUDIO_DIR / f"{slug}.mp3")

    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "mp3",
        "-o", str(out_mp3),
        url,
    ]
    subprocess.run(cmd, check=True)

    if not out_mp3.exists():
        # fallback if yt-dlp adds extension variants
        candidates = list(AUDIO_DIR.glob(f"{slug}*.mp3"))
        if not candidates:
            raise RuntimeError("Audio not found after download.")
        candidates.sort(key=lambda p: (p.stem != slug, -p.stat().st_mtime))
        out_mp3 = candidates[0]

    episode_id = out_mp3.stem
    return episode_id, out_mp3

def copy_local_audio(src_path: str) -> tuple[str, Path]:
    """Copy a local file keeping readable filename."""
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    slug = _slugify(src.stem)
    dest = _unique_path(AUDIO_DIR / f"{slug}{src.suffix.lower()}")
    shutil.copy2(src, dest)
    return dest.stem, dest
