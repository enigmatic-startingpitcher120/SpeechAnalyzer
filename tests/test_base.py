def test_imports():
    from analyzers.base import Segment, TranscriptDoc, BaseAnalyzer, AnalyzerResult, SUPPORTED_LANGUAGES
    assert "de" in SUPPORTED_LANGUAGES
    assert "en" in SUPPORTED_LANGUAGES

def test_segment_defaults():
    from analyzers.base import Segment
    s = Segment(start=0.0, end=5.0, text="hello")
    assert s.confidence == 1.0

def test_transcript_doc_annotations_default(nlp):
    from analyzers.base import Segment, TranscriptDoc
    seg = Segment(0.0, 5.0, "hello world", 0.9)
    spacy_doc = nlp("hello world")
    doc = TranscriptDoc(
        raw_text="hello world",
        clean_text="hello world",
        segments=[seg],
        spacy_doc=spacy_doc,
        language="en",
    )
    assert doc.annotations == {}

def test_analyzer_can_run_blocks_unsupported_language(nlp):
    from analyzers.base import BaseAnalyzer, TranscriptDoc, Segment

    class POSAnalyzer(BaseAnalyzer):
        name = "test_pos"
        requires_pos = True
        def run(self, doc): ...

    seg = Segment(0.0, 1.0, "text", 1.0)
    doc = TranscriptDoc("text", "text", [seg], nlp("text"), "zh")
    assert not POSAnalyzer().can_run(doc)

def test_analyzer_can_run_allows_supported_language(nlp):
    from analyzers.base import BaseAnalyzer, TranscriptDoc, Segment

    class POSAnalyzer(BaseAnalyzer):
        name = "test_pos"
        requires_pos = True
        def run(self, doc): ...

    seg = Segment(0.0, 1.0, "text", 1.0)
    doc = TranscriptDoc("text", "text", [seg], nlp("text"), "en")
    assert POSAnalyzer().can_run(doc)

def test_analyzer_no_pos_runs_on_any_language(nlp):
    from analyzers.base import BaseAnalyzer, TranscriptDoc, Segment

    class SimpleAnalyzer(BaseAnalyzer):
        name = "test_simple"
        requires_pos = False
        def run(self, doc): ...

    seg = Segment(0.0, 1.0, "text", 1.0)
    doc = TranscriptDoc("text", "text", [seg], nlp("text"), "zh")
    assert SimpleAnalyzer().can_run(doc)
