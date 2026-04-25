from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import spacy.tokens

SUPPORTED_LANGUAGES = {"de", "en", "fr", "es", "it", "nl", "pt"}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float = 1.0


@dataclass
class TranscriptDoc:
    raw_text: str
    clean_text: str
    segments: list[Segment]
    spacy_doc: spacy.tokens.Doc
    language: str
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzerResult:
    name: str
    metrics: dict[str, Any]
    figures: list[Any]
    summary: str
    warnings: list[str] = field(default_factory=list)


class BaseAnalyzer:
    name: str = ""
    requires_pos: bool = False

    def can_run(self, doc: TranscriptDoc) -> bool:
        if self.requires_pos and doc.language not in SUPPORTED_LANGUAGES:
            return False
        return True

    def run(self, doc: TranscriptDoc) -> AnalyzerResult:
        raise NotImplementedError
