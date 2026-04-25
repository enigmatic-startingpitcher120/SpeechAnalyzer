import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV"}


def _get_lemmas(spacy_doc):
    return [
        t.lemma_.lower()
        for t in spacy_doc
        if not t.is_punct and not t.is_space and t.lemma_.strip()
    ]


def _brunet(tokens: list, lemmas: list) -> float:
    n = len(tokens)
    v = len(set(lemmas))
    if n == 0 or v == 0:
        return 0.0
    return n ** (v ** -0.165)


def _honore(tokens: list, lemmas: list) -> float:
    n = len(tokens)
    v = len(set(lemmas))
    if n == 0 or v == 0:
        return 0.0
    counts = Counter(lemmas)
    f1 = sum(1 for c in counts.values() if c == 1)
    if f1 == v:
        return 0.0
    return 100 * math.log(n) / (1 - f1 / v)


def _lexical_density(spacy_doc) -> float:
    tokens = [t for t in spacy_doc if not t.is_space and not t.is_punct]
    if not tokens:
        return 0.0
    content = sum(1 for t in tokens if t.pos_ in CONTENT_POS)
    return content / len(tokens)


class ComplexityAnalyzer(BaseAnalyzer):
    name = "complexity"
    requires_pos = True

    def run(self, doc) -> AnalyzerResult:
        tokens_all = [t for t in doc.spacy_doc if not t.is_space and not t.is_punct]
        lemmas = _get_lemmas(doc.spacy_doc)
        token_texts = [t.text.lower() for t in tokens_all]

        brunet = _brunet(token_texts, lemmas)
        honore = _honore(token_texts, lemmas)
        lex_density = _lexical_density(doc.spacy_doc)

        metrics = {
            "brunet_index": round(brunet, 4),
            "honore_index": round(honore, 4),
            "lexical_density": round(lex_density, 4),
        }

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.set_label("complexity_metrics")
        ax.bar(
            ["Lexikalische Dichte"],
            [lex_density],
            color="#4C72B0",
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Wert")
        ax.set_title(
            f"Komplexität  |  Brunet={brunet:.2f}  |  Honoré={honore:.1f}"
        )
        fig.tight_layout()

        summary = (
            f"Brunet-Index={brunet:.2f}, Honoré-Index={honore:.1f}, "
            f"Lexikalische Dichte={lex_density:.3f}"
        )
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig], summary=summary)
