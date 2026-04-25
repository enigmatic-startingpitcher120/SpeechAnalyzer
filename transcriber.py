import whisper
from analyzers.base import Segment


def transcribe(video_path: str, model_name: str = "base") -> tuple[list[Segment], str]:
    model = whisper.load_model(model_name)
    result = model.transcribe(video_path)

    segments = []
    for seg in result["segments"]:
        logprob = seg.get("avg_logprob", 0.0)
        confidence = min(1.0, max(0.0, 1.0 + logprob / 5.0))
        segments.append(Segment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=seg["text"].strip(),
            confidence=confidence,
        ))

    language = result.get("language", "unknown")
    return segments, language
