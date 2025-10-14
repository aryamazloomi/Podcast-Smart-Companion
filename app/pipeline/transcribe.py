# app/pipeline/transcribe.py
from faster_whisper import WhisperModel
from pathlib import Path
import json

TRANSCRIPT_DIR = Path("data/transcripts")
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# model_size: "small", "medium", "large-v3" (choose per machine)
_asr_model = WhisperModel("small", device="cuda" if True else "cpu", compute_type="float16")

def transcribe_episode(episode_id: str, audio_path: Path, language: str | None = None) -> Path:
    segments, info = _asr_model.transcribe(str(audio_path), language=language, vad_filter=True)
    rec = {
        "episode_id": episode_id,
        "language": info.language,
        "duration": info.duration,
        "segments": [
            {
                "id": i,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            } for i, seg in enumerate(segments)
        ]
    }
    out = TRANSCRIPT_DIR / f"{episode_id}.json"
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
