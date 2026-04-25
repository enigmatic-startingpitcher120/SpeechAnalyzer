from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult


def _get_lemmas(spacy_doc):
    return [
        t.lemma_.lower()
        for t in spacy_doc
        if not t.is_stop and not t.is_punct and not t.is_space and t.lemma_.strip()
    ]


def _ttr(lemmas: list) -> float:
    if not lemmas:
        return 0.0
    return len(set(lemmas)) / len(lemmas)


def _mattr(lemmas: list, window_size: int = 50) -> float:
    if not lemmas:
        return 0.0
    if len(lemmas) <= window_size:
        return len(set(lemmas)) / len(lemmas)
    ttrs = [
        len(set(lemmas[i : i + window_size])) / window_size
        for i in range(len(lemmas) - window_size + 1)
    ]
    return sum(ttrs) / len(ttrs)


def _chao1(lemmas: list) -> tuple:
    if not lemmas:
        return 0.0, 0, 0, []
    counts = Counter(lemmas)
    s_obs = len(counts)
    f1 = sum(1 for c in counts.values() if c == 1)
    f2 = sum(1 for c in counts.values() if c == 2)
    warnings = []
    if f2 == 0:
        f2 = 1
        warnings.append("Chao1: f2=0, Smoothing angewendet (f2=1)")
    estimate = s_obs + (f1 ** 2) / (2 * f2)
    return estimate, f1, s_obs, warnings


class VocabularyAnalyzer(BaseAnalyzer):
    name = "vocabulary"
    requires_pos = True

    def run(self, doc) -> AnalyzerResult:
        lemmas = _get_lemmas(doc.spacy_doc)
        ttr = _ttr(lemmas)
        mattr = _mattr(lemmas)
        chao1, f1, s_obs, warns = _chao1(lemmas)

        metrics = {
            "ttr": round(ttr, 4),
            "mattr": round(mattr, 4),
            "chao1": round(chao1, 2),
            "observed_types": s_obs,
            "hapax_legomena": f1,
            "total_lemmas": len(lemmas),
        }

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.set_label("vocabulary_metrics")
        axes[0].bar(["TTR", "MATTR"], [ttr, mattr], color=["#4C72B0", "#DD8452"])
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Type-Token-Ratio")
        axes[0].set_ylabel("Wert")
        axes[1].bar(["Beobachtet", "Chao1-Schätzung"], [s_obs, chao1], color=["#4C72B0", "#DD8452"])
        axes[1].set_title("Wortschatzschätzung")
        axes[1].set_ylabel("Anzahl Typen")
        fig.tight_layout()

        summary = (
            f"Wortschatz: MATTR={mattr:.3f}, TTR={ttr:.3f}, "
            f"Chao1-Schätzung={chao1:.0f} (beobachtet: {s_obs})"
        )
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig], summary=summary, warnings=warns)
