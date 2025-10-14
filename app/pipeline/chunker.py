# app/pipeline/chunker.py
import json
from pathlib import Path
import pandas as pd

CHUNK_DIR = Path("data/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

def make_chunks(transcript_json: Path, max_chars=1000, overlap_chars=150) -> Path:
    doc = json.loads(transcript_json.read_text(encoding="utf-8"))
    episode_id = doc["episode_id"]
    chunks = []
    buf, ts_start, ts_end = "", None, None

    for seg in doc["segments"]:
        text = seg["text"]
        if not buf:
            ts_start = seg["start"]
        candidate = (buf + " " + text).strip() if buf else text
        if len(candidate) <= max_chars:
            buf = candidate
            ts_end = seg["end"]
        else:
            # flush
            chunks.append({"episode_id": episode_id, "text": buf, "ts_start": ts_start, "ts_end": ts_end})
            # start new with overlap from end of buf
            overlap = buf[-overlap_chars:] if len(buf) > overlap_chars else buf
            buf = (overlap + " " + text).strip()
            ts_start = max(ts_end - 5, 0) if ts_end else seg["start"]
            ts_end = seg["end"]

    if buf:
        chunks.append({"episode_id": episode_id, "text": buf, "ts_start": ts_start, "ts_end": ts_end})

    df = pd.DataFrame(chunks).reset_index().rename(columns={"index":"chunk_id"})
    out = CHUNK_DIR / f"{episode_id}.parquet"
    df.to_parquet(out, index=False)
    return out
