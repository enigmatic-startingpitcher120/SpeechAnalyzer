# tests/test_complexity.py
import math

def test_complexity_returns_result(simple_doc):
    from analyzers.complexity import ComplexityAnalyzer
    result = ComplexityAnalyzer().run(simple_doc)
    assert result.name == "complexity"
    assert "brunet_index" in result.metrics
    assert "honore_index" in result.metrics
    assert "lexical_density" in result.metrics


def test_brunet_index_positive(simple_doc):
    from analyzers.complexity import ComplexityAnalyzer
    result = ComplexityAnalyzer().run(simple_doc)
    assert result.metrics["brunet_index"] > 0


def test_lexical_density_between_zero_and_one(simple_doc):
    from analyzers.complexity import ComplexityAnalyzer
    result = ComplexityAnalyzer().run(simple_doc)
    assert 0.0 <= result.metrics["lexical_density"] <= 1.0


def test_complexity_requires_pos():
    from analyzers.complexity import ComplexityAnalyzer
    assert ComplexityAnalyzer().requires_pos is True


def test_complexity_produces_figure(simple_doc):
    from analyzers.complexity import ComplexityAnalyzer
    result = ComplexityAnalyzer().run(simple_doc)
    assert len(result.figures) >= 1


def test_honore_zero_when_all_hapax(nlp):
    from analyzers.base import Segment, TranscriptDoc
    from analyzers.complexity import ComplexityAnalyzer
    # Text where every lemma is unique → f1 == V → denominator = 0
    text = "alpha beta gamma delta epsilon"
    spacy_doc = nlp(text)
    doc = TranscriptDoc(text, text, [Segment(0.0, 5.0, text, 1.0)], spacy_doc, "en")
    result = ComplexityAnalyzer().run(doc)
    assert result.metrics["honore_index"] == 0.0
