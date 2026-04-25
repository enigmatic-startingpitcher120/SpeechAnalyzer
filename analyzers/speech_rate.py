import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyzers.base import BaseAnalyzer, AnalyzerResult

CONFIDENCE_THRESHOLD = 0.5


def _tokens(spacy_doc) -> list:
    return [t for t in spacy_doc if not t.is_space and not t.is_punct]


def _wpm_net(segments: list, token_count: int) -> tuple:
    speech_seconds = sum(
        s.end - s.start for s in segments if s.confidence >= CONFIDENCE_THRESHOLD
    )
    if speech_seconds < 1e-9:
        return 0.0, 0.0
    return token_count / (speech_seconds / 60), speech_seconds


def _wpm_gross(segments: list, token_count: int) -> tuple:
    if not segments:
        return 0.0, 0.0
    duration = segments[-1].end - segments[0].start
    if duration == 0:
        return 0.0, 0.0
    return token_count / (duration / 60), duration


class SpeechRateAnalyzer(BaseAnalyzer):
    name = "speech_rate"
    requires_pos = False

    def run(self, doc) -> AnalyzerResult:
        token_count = len(_tokens(doc.spacy_doc))
        wpm_net, net_seconds = _wpm_net(doc.segments, token_count)
        wpm_gross, gross_seconds = _wpm_gross(doc.segments, token_count)

        metrics = {
            "wpm_net": round(wpm_net, 1),
            "wpm_gross": round(wpm_gross, 1),
            "total_tokens": token_count,
            "net_speech_seconds": round(net_seconds, 1),
            "gross_duration_seconds": round(gross_seconds, 1),
        }

        fig, ax = plt.subplots(figsize=(6, 4))
        fig.set_label("speech_rate")
        ax.bar(["Netto-WPM", "Brutto-WPM"], [wpm_net, wpm_gross], color=["#4C72B0", "#DD8452"])
        ax.set_ylabel("Wörter pro Minute")
        ax.set_title("Redegeschwindigkeit")
        fig.tight_layout()

        summary = f"Redegeschwindigkeit: Netto={wpm_net:.0f} WPM, Brutto={wpm_gross:.0f} WPM"
        return AnalyzerResult(name=self.name, metrics=metrics, figures=[fig], summary=summary)
