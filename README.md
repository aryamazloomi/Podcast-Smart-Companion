# 🎧 Podcast Smart Companion

An AI-powered assistant that turns podcast audio or YouTube videos into structured, searchable insights.  
It automatically **transcribes**, **summarizes**, **extracts highlights**, and **recommends similar episodes** based on **semantic meaning** and **emotional tone**.

---

## 🚀 Features

- 🎙️ **Audio & YouTube Input**  
  Upload audio files (`.mp3`, `.wav`, `.m4a`) or paste any YouTube URL.

- 🧠 **Speech-to-Text (ASR)**  
  Uses [Whisper (faster-whisper)](https://github.com/guillaumekln/faster-whisper) for fast and accurate transcription.

- ✂️ **Smart Chunking**  
  Splits long transcripts into manageable text segments with timestamp tracking.

- 🔍 **Semantic Embeddings**  
  Converts each text chunk into vector embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).

- 🪶 **Summarization**  
  Generates concise TL;DR summaries and bullet highlights using Hugging Face transformer models (`bart-large-cnn` or `distilbart-cnn-12-6`).

- ❤️ **Emotion & Tone Analysis**  
  Detects the overall emotional tone (joy, sadness, anger, etc.) via `j-hartmann/emotion-english-distilroberta-base`.

- 🏷️ **Keyword & Topic Extraction**  
  Extracts main ideas and topics via KeyBERT or BERTopic.

- 🤖 **Recommendations**  
  Suggests similar episodes by combining semantic similarity and emotional profiles.

- 📊 **Streamlit Dashboard**  
  A clean web UI to upload, process, explore, and search episodes.

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend / UI** | Streamlit |
| **Transcription** | faster-whisper (CTranslate2) |
| **Summarization & Emotion** | Hugging Face Transformers |
| **Embeddings** | Sentence Transformers |
| **Vector Storage** | ChromaDB (local persistent) |
| **Database** | SQLite (via SQLModel) |
| **Visualization** | Plotly |
| **Packaging** | Python 3.10+ |

---

## ⚙️ Local Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/aryamazloomi/podcast-smart-companion.git
cd podcast-smart-companion
```


### 2️⃣ Create a Virtual Environment
```bash
python -m venv .podcast
# Windows
.podcast\Scripts\activate
# macOS/Linux
source .podcast/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 4️⃣ Run the App
```bash
streamlit run app/main.py
```

### How It Works (Pipeline)

Audio/YouTube -> Whisper (Transcription)
             -> Chunking (Timestamps)
             -> Embeddings (Semantic Vectors)
             -> Summarization (TL;DR + Bullets)
             -> Emotion Analysis
             -> Keyword Extraction
             -> Chroma + SQLite Storage
             -> Streamlit UI + Recommendations


## Model Sources (Hugging Face)

| Purpose       | Model                                           | HF Link                                                                              |
| ------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------ |
| Transcription | `faster-whisper`                                | [CTranslate2/Whisper](https://github.com/guillaumekln/faster-whisper)                |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2`        | [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)        |
| Summarization | `facebook/bart-large-cnn`                       | [Hugging Face](https://huggingface.co/facebook/bart-large-cnn)                       |
| Emotion       | `j-hartmann/emotion-english-distilroberta-base` | [Hugging Face](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) |


## Author
Arya Mazloomi
AI Engineer & Full-Stack Developer
- Toronto, Canada
