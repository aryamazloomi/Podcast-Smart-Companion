# app/pipeline/embedder.py
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

_client = chromadb.PersistentClient(path=".db/chroma")
_collection = _client.get_or_create_collection(
    name="chunks",
    metadata={"hnsw:space": "cosine"},
    embedding_function=None  # we pass vectors directly
)

_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def index_chunks(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    texts = df["text"].tolist()
    ids = [f"{r.episode_id}::{r.chunk_id}" for r in df.itertuples()]
    metas = [
        {"episode_id": r.episode_id, "chunk_id": int(r.chunk_id), "ts_start": float(r.ts_start),
         "ts_end": float(r.ts_end), "preview": r.text[:200]} for r in df.itertuples()
    ]
    embs = _model.encode(texts, normalize_embeddings=True).tolist()
    _collection.add(ids=ids, metadatas=metas, embeddings=embs, documents=texts)
    return len(ids)

def query_similar(text: str, n: int = 5):
    q = _model.encode([text], normalize_embeddings=True).tolist()[0]
    res = _collection.query(query_embeddings=[q], n_results=n)
    return res
