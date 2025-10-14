# app/pipeline/__init__.py (optional convenience)
from .ingest import download_youtube, copy_local_audio
from .transcribe import transcribe_episode
from .chunker import make_chunks
from .embedder import index_chunks
from .summarize import map_reduce_summary
from .emotion import emotion_profile
from .keywords import extract_keywords
from .cluster import cluster_chunks
from .recommend import recommend_for_episode

def process_episode_from_youtube(url: str, chunk_chars=1000):
    eid, path = download_youtube(url)
    t = transcribe_episode(eid, path)
    c = make_chunks(t, max_chars=chunk_chars)
    n = index_chunks(c)
    s = map_reduce_summary(eid, c)
    e = emotion_profile(eid, c)
    k = extract_keywords(eid, c)
    cl = cluster_chunks(eid, c, k=6)
    return eid
