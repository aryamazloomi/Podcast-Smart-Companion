# app/pipeline/recommend.py
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import chromadb
from sentence_transformers import SentenceTransformer
import sqlite3

_client = chromadb.PersistentClient(path=".db/chroma")
_chunks = _client.get_or_create_collection("chunks")
_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _episode_centroid(episode_id: str):
    res = _chunks.get(where={"episode_id": episode_id}, include=["embeddings","metadatas"])
    embs = res.get("embeddings", [])
    if embs is None or len(embs) == 0:
        return None
    M = np.array(embs, dtype=np.float32)
    return M.mean(axis=0)

def _emotion_vec(episode_id: str) -> np.ndarray | None:
    p = Path(f"data/features/{episode_id}.json")
    if not p.exists(): return None
    obj = json.loads(p.read_text())
    labels = obj["emotion_labels"]
    vec = np.array([obj["mean_profile"][lab] for lab in labels])
    return vec / (np.linalg.norm(vec)+1e-9)

def recommend_for_episode(episode_id: str, topk=5, alpha=0.8):
    # alpha weights semantic similarity; (1-alpha) weights emotion similarity
    # 1) compute episode centroid
    q = _episode_centroid(episode_id)
    if q is None: return []
    # 2) list all other episode ids
    # naive scan: read from chunk metadatas
    all_meta = _chunks.get(include=["metadatas"])
    others = sorted({m["episode_id"] for m in all_meta["metadatas"] if m["episode_id"] != episode_id})

    sims = []
    qe = _emotion_vec(episode_id)
    for other in others:
        c = _episode_centroid(other)
        if c is None: continue
        sem = 1 - cdist(q[None,:], c[None,:], metric="cosine")[0,0]
        score = sem
        if qe is not None:
            oe = _emotion_vec(other)
            if oe is not None and len(qe)==len(oe):
                emo = 1 - cdist(qe[None,:], oe[None,:], metric="cosine")[0,0]
                score = alpha*sem + (1-alpha)*emo
        sims.append((other, float(score), float(sem)))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topk]
