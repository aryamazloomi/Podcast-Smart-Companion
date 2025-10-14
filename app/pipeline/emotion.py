# app/pipeline/emotion.py
from transformers import pipeline
import pandas as pd
from pathlib import Path
import json
import numpy as np

FEAT_DIR = Path("data/features")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Use top_k=None instead of return_all_scores=True
_emotion = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,              # returns all labels with scores
    device_map="auto"
)

def emotion_profile(episode_id: str, chunks_parquet: Path) -> Path:
    df = pd.read_parquet(chunks_parquet)
    # Output shape: List[List[{"label":..., "score":...}, ...]]
    scores = _emotion(df["text"].tolist())

    labels = [d["label"] for d in scores[0]]           # label order
    mat = np.array([[d["score"] for d in row] for row in scores], dtype=np.float32)

    mean_profile = dict(zip(labels, mat.mean(axis=0).tolist()))
    out = {"episode_id": episode_id, "emotion_labels": labels, "mean_profile": mean_profile}
    f = FEAT_DIR / f"{episode_id}.json"
    f.write_text(json.dumps(out, indent=2))
    return f
