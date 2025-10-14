# app/main.py
import streamlit as st
from pathlib import Path
from pipeline import ingest, transcribe, chunker, embedder, summarize, emotion, keywords, cluster, recommend
import json
import pandas as pd

st.set_page_config(page_title="Podcast Smart Companion", layout="wide")

st.sidebar.title("Podcast Smart Companion")
mode = st.sidebar.radio("Mode", ["Upload/Process", "Episode Viewer", "Search & Recommend", "Library", "Settings"])

if mode == "Upload/Process":
    st.header("Upload or paste a YouTube link")
    yt = st.text_input("YouTube URL")
    up = st.file_uploader("Or upload audio", type=["mp3","m4a","wav"])
    colA, colB = st.columns(2)
    with colA:
        asr_size = st.selectbox("ASR model size", ["small","medium","large-v3"], index=0)
    with colB:
        chunk_chars = st.slider("Chunk size (chars)", 300, 2000, 1000, 50)

    if st.button("Process Episode", type="primary"):
        try:
            if yt:
                episode_id, audio_path = ingest.download_youtube(yt)
                title = "YouTube Episode"
            elif up is not None:
                tmp = Path("data/tmp"); tmp.mkdir(parents=True, exist_ok=True)
                local = tmp / up.name; local.write_bytes(up.read())
                episode_id, audio_path = ingest.copy_local_audio(str(local))
                title = up.name
            else:
                st.warning("Provide a URL or upload a file"); st.stop()

            st.info("Transcribing...")
            trans_path = transcribe.transcribe_episode(episode_id, audio_path)

            st.info("Chunking...")
            chunks_path = chunker.make_chunks(trans_path, max_chars=chunk_chars)

            st.info("Indexing embeddings...")
            n = embedder.index_chunks(chunks_path)
            st.success(f"Indexed {n} chunks")

            st.info("Summarizing...")
            sum_path = summarize.map_reduce_summary(episode_id, chunks_path)

            st.info("Emotion profile...")
            emo_path = emotion.emotion_profile(episode_id, chunks_path)

            st.info("Keywords...")
            kw = keywords.extract_keywords(episode_id, chunks_path)

            st.info("Clustering...")
            cinfo = cluster.cluster_chunks(episode_id, chunks_path, k=6)

            st.success(f"Done: {episode_id}")
            st.json({"episode_id": episode_id, "title": title})
        except Exception as e:
            st.error(str(e))

elif mode == "Episode Viewer":
    st.header("Episode Viewer")
    # simple picker by listing chunk files
    eps = [p.stem for p in Path("data/chunks").glob("*.parquet")]
    episode_id = st.selectbox("Select episode", eps)
    if episode_id:
        with st.expander("Summary"):
            sfile = Path(f"data/summaries/{episode_id}.json")
            if sfile.exists():
                sj = json.loads(sfile.read_text())
                st.subheader("TL;DR")
                st.write(sj["tl_dr"])
                st.subheader("Bullets")
                for b in sj["bullets"]:
                    st.markdown(f"- {b}")

        df = pd.read_parquet(f"data/chunks/{episode_id}.parquet")
        st.subheader("Timeline (chunks)")
        st.dataframe(df[["chunk_id","ts_start","ts_end","text"]])

        ffile = Path(f"data/features/{episode_id}.json")
        if ffile.exists():
            fj = json.loads(ffile.read_text())
            st.subheader("Emotion profile (mean)")
            st.json(fj["mean_profile"])

elif mode == "Search & Recommend":
    st.header("Search")
    q = st.text_input("Semantic search query")
    if st.button("Search"):
        if q.strip():
            res = embedder.query_similar(q, n=8)
            st.write(res)

    st.header("Recommendations")
    eps = [p.stem for p in Path("data/chunks").glob("*.parquet")]
    eid = st.selectbox("Episode", eps)
    alpha = st.slider("Semantic weight (alpha)", 0.0, 1.0, 0.8, 0.05)
    if st.button("Get Recommendations"):
        recs = recommend.recommend_for_episode(eid, topk=6, alpha=alpha)
        for other, score, sem in recs:
            st.markdown(f"**{other}** — score: {score:.3f} (semantic {sem:.3f})")

elif mode == "Library":
    st.header("Your Library")
    eps = [p.stem for p in Path("data/chunks").glob("*.parquet")]
    st.write(pd.DataFrame({"episode_id": eps}))

else:
    st.header("Settings")
    st.write("Configure paths, models, and performance options here.")
