def test_imports():
    from analyzers.base import Segment, TranscriptDoc, BaseAnalyzer, AnalyzerResult, SUPPORTED_LANGUAGES
    assert "de" in SUPPORTED_LANGUAGES
    assert "en" in SUPPORTED_LANGUAGES
