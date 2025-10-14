# app/pipeline/summarize.py
from transformers import pipeline
import pandas as pd
from pathlib import Path
import json

SUM_DIR = Path("data/summaries")
SUM_DIR.mkdir(parents=True, exist_ok=True)

_summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")

def _batch_summarize(texts, max_len=200, min_len=60):
    outs = []
    for t in texts:
        t = t[:3500]  # safety
        s = _summarizer(t, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        outs.append(s)
    return outs

def map_reduce_summary(episode_id: str, chunks_parquet: Path) -> Path:
    df = pd.read_parquet(chunks_parquet)
    partials = _batch_summarize(df["text"].tolist())
    combined = "\n".join(partials)
    tl_dr = _batch_summarize([combined], max_len=180, min_len=80)[0]
    bullets = [s for s in partials]
    out = {
        "episode_id": episode_id,
        "tl_dr": tl_dr,
        "bullets": bullets[:10]
    }
    file = SUM_DIR / f"{episode_id}.json"
    file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return file
