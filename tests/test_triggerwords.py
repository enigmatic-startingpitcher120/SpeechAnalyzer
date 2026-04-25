from pathlib import Path


def test_triggerwords_returns_result(simple_doc):
    from analyzers.triggerwords import TriggerwordsAnalyzer
    result = TriggerwordsAnalyzer().run(simple_doc)
    assert result.name == "triggerwords"
    assert "wealth" in result.metrics
    assert "drama" in result.metrics
    assert "hype" in result.metrics
    assert "self_ref" in result.metrics


def test_triggerwords_counts_are_non_negative(simple_doc):
    from analyzers.triggerwords import TriggerwordsAnalyzer
    result = TriggerwordsAnalyzer().run(simple_doc)
    for cat in result.metrics.values():
        assert cat["count"] >= 0
        assert 0.0 <= cat["percent"] <= 100.0


def test_triggerwords_matches_on_lemma_level(nlp):
    from analyzers.base import Segment, TranscriptDoc
    from analyzers.triggerwords import TriggerwordsAnalyzer

    text = "ich liebe es und mich freut es auch"
    spacy_doc = nlp(text)
    doc = TranscriptDoc(text, text, [Segment(0.0, 3.0, text, 1.0)], spacy_doc, "en",
                        annotations={"nlp": nlp})
    result = TriggerwordsAnalyzer().run(doc)
    assert result.metrics["self_ref"]["count"] >= 1


def test_triggerwords_requires_pos():
    from analyzers.triggerwords import TriggerwordsAnalyzer
    assert TriggerwordsAnalyzer().requires_pos is True


def test_triggerwords_produces_figure(simple_doc):
    from analyzers.triggerwords import TriggerwordsAnalyzer
    result = TriggerwordsAnalyzer().run(simple_doc)
    assert len(result.figures) >= 1


def test_wordlists_exist():
    base = Path("data/triggerwords")
    for name in ["wealth.txt", "drama.txt", "hype.txt", "self_ref.txt"]:
        assert (base / name).exists(), f"Missing: {base / name}"
