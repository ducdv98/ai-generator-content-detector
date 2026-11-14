"""Evidence model shared by all detector layers, and the verdict logic."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# What a finding points to.
AI = "ai"
NOT_AI = "not_ai"
NEUTRAL = "neutral"

# How much weight a finding carries.
DECLARED = "declared"        # cryptographically signed or explicit machine-readable tag
STRONG = "strong"            # known watermark or generator-specific metadata
STATISTICAL = "statistical"  # classifier probability
WEAK = "weak"                # easily forged or circumstantial
INFO = "info"                # nothing found / layer unavailable


@dataclass
class Evidence:
    layer: str      # "provenance" | "watermark" | "classifier"
    check: str      # short name of the check
    finding: str    # one-sentence, human-readable result
    points_to: str = NEUTRAL
    strength: str = INFO
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    verdict: str
    confidence: str  # "declared" | "high" | "probabilistic" | "none"
    summary: str
    evidence: list[Evidence]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_report(evidence: list[Evidence]) -> Report:
    """Combine evidence, strongest kind first. Order matters: a signed
    declaration beats a watermark, which beats a classifier guess."""
    ai = [e for e in evidence if e.points_to == AI]

    declared = [e for e in ai if e.strength == DECLARED]
    if declared:
        return Report("AI-generated", "declared",
                      f"The file itself declares it: {declared[0].finding}", evidence)

    strong = [e for e in ai if e.strength == STRONG]
    if strong:
        return Report("AI-generated", "high", strong[0].finding, evidence)

    stat = [e for e in evidence if e.strength == STATISTICAL]
    if stat:
        e = stat[0]
        p = e.details.get("ai_probability")
        if e.points_to == AI:
            return Report("Likely AI-generated", "probabilistic",
                          f"No tag or watermark found; the classifier estimates {p:.0%} AI. "
                          "This is a statistical guess and can be wrong.", evidence)
        if e.points_to == NOT_AI:
            return Report("Likely not AI-generated", "probabilistic",
                          f"No tag or watermark found; the classifier estimates {p:.0%} AI. "
                          "This is a statistical guess and can be wrong.", evidence)
        return Report("Inconclusive", "none",
                      f"No tag or watermark found, and the classifier is undecided ({p:.0%} AI).",
                      evidence)

    if any(e.points_to == AI and e.strength == WEAK for e in evidence):
        weak = next(e for e in evidence if e.points_to == AI and e.strength == WEAK)
        return Report("Possibly AI-generated", "none",
                      f"Only weak, easily forged evidence: {weak.finding}", evidence)

    return Report("No evidence of AI generation", "none",
                  "No AI tag, known watermark or classifier signal was found. This does not "
                  "prove the content is real: metadata is easily stripped and most generators "
                  "leave no detectable mark.", evidence)
