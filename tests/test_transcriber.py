# tests/test_transcriber.py

def test_transcribe_returns_segments_and_language(monkeypatch):
    mock_result = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 3.0, "text": " Hello world", "avg_logprob": -0.3},
            {"start": 3.0, "end": 6.0, "text": " This is a test", "avg_logprob": -0.8},
        ],
    }

    class MockModel:
        def transcribe(self, path, **kwargs):
            return mock_result

    monkeypatch.setattr("whisper.load_model", lambda name: MockModel())
    from transcriber import transcribe

    segments, lang = transcribe("fake.mp4")

    assert lang == "en"
    assert len(segments) == 2
    assert segments[0].text == "Hello world"
    assert segments[1].text == "This is a test"
    assert 0.0 <= segments[0].confidence <= 1.0
    assert 0.0 <= segments[1].confidence <= 1.0


def test_transcribe_confidence_lower_for_bad_logprob(monkeypatch):
    mock_result = {
        "language": "de",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": " gut", "avg_logprob": -0.1},
            {"start": 2.0, "end": 4.0, "text": " schlecht", "avg_logprob": -4.0},
        ],
    }

    class MockModel:
        def transcribe(self, path, **kwargs):
            return mock_result

    monkeypatch.setattr("whisper.load_model", lambda name: MockModel())
    from transcriber import transcribe

    segments, _ = transcribe("fake.mp4")
    assert segments[0].confidence > segments[1].confidence


def test_transcribe_strips_whitespace(monkeypatch):
    mock_result = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "  spaces  ", "avg_logprob": 0.0}],
    }

    class MockModel:
        def transcribe(self, path, **kwargs):
            return mock_result

    monkeypatch.setattr("whisper.load_model", lambda name: MockModel())
    from transcriber import transcribe

    segments, _ = transcribe("fake.mp4")
    assert segments[0].text == "spaces"
