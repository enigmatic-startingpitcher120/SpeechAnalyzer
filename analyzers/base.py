from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    segments: list
    spacy_doc: object
    language: str
    annotations: dict = field(default_factory=dict)


@dataclass
class AnalyzerResult:
    name: str
    metrics: dict
    figures: list
    summary: str
    warnings: list = field(default_factory=list)


class BaseAnalyzer:
    name: str = ""
    requires_pos: bool = False

    def can_run(self, doc: TranscriptDoc) -> bool:
        if self.requires_pos and doc.language not in SUPPORTED_LANGUAGES:
            return False
        return True

    def run(self, doc: TranscriptDoc) -> AnalyzerResult:
        raise NotImplementedError
