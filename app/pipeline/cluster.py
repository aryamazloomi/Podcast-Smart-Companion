# app/pipeline/cluster.py
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from pathlib import Path
import json
import numpy as np

def cluster_chunks(episode_id: str, chunks_parquet: Path, k=6) -> dict:
    df = pd.read_parquet(chunks_parquet)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(df["text"].tolist(), normalize_embeddings=True)
    kmeans = KMeans(n_clusters=k, n_init="auto").fit(embs)
    df["cluster"] = kmeans.labels_
    labels = []
    for c in range(k):
        idx = np.where(kmeans.labels_ == c)[0][:3]
        preview = " | ".join(df.iloc[idx]["text"].str[:120].tolist())
        labels.append({"cluster": int(c), "preview": preview})
    return {"episode_id": episode_id, "k": k, "labels": labels}
