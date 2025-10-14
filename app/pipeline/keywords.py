# app/pipeline/keywords.py
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import json
from collections import Counter

_kw_model = KeyBERT(model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))

def extract_keywords(episode_id: str, chunks_parquet: Path, topk=5) -> dict:
    df = pd.read_parquet(chunks_parquet)
    agg = Counter()
    for t in df["text"].tolist():
        pairs = _kw_model.extract_keywords(t, keyphrase_ngram_range=(1,2), top_n=topk, stop_words="english")
        for kw, score in pairs:
            agg[kw] += score
    top = [{"keyword": k, "score": float(v)} for k, v in agg.most_common(20)]
    return {"episode_id": episode_id, "keywords": top}
