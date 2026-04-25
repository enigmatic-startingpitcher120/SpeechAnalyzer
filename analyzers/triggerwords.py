from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult

DATA_DIR = Path(__file__).parent.parent / "data" / "triggerwords"

CATEGORIES = {
    "wealth": DATA_DIR / "wealth.txt",
    "drama": DATA_DIR / "drama.txt",
    "hype": DATA_DIR / "hype.txt",
    "self_ref": DATA_DIR / "self_ref.txt",
}


def _load_and_lemmatize_wordlist(path: Path, nlp) -> set:
    lemmas: set = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word:
                for token in nlp(word):
                    lemmas.add(token.lemma_.lower())
    return lemmas


class TriggerwordsAnalyzer(BaseAnalyzer):
    name = "triggerwords"
    requires_pos = True

    def run(self, doc) -> AnalyzerResult:
        nlp = doc.annotations.get("nlp")
        doc_lemmas = [
            t.lemma_.lower()
            for t in doc.spacy_doc
            if not t.is_space and not t.is_punct
        ]
        total = len(doc_lemmas) or 1

        metrics: dict = {}
        for cat, path in CATEGORIES.items():
            wordlist = _load_and_lemmatize_wordlist(path, nlp)
            count = sum(1 for lemma in doc_lemmas if lemma in wordlist)
            metrics[cat] = {
                "count": count,
                "percent": round(count / total * 100, 2),
            }

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.set_label("triggerwords")
        categories = list(metrics.keys())
        counts = [metrics[c]["count"] for c in categories]
        labels = ["Reichtum", "Drama", "Hype", "Ich-Bezug"]
        ax.bar(labels, counts, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
        ax.set_ylabel("Trefferanzahl")
        ax.set_title("Trigger-Wort-Analyse")
        fig.tight_layout()

        summary_parts = [f"{k}={v['count']}({v['percent']}%)" for k, v in metrics.items()]
        summary = "Triggerworte: " + ", ".join(summary_parts)
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig], summary=summary)
