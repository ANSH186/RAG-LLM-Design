"""Prompt-injection and banned-contract-term security rules.

The detector is conservative about ordinary business language. It looks for directives
aimed at an AI/automation/reviewer or claims of an override/prior approval. It does not
treat harmless phrases such as "AI-powered product", "ERP system", "system integration
team" or "assistant manager" as injections.
"""
from __future__ import annotations

import re
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from src.utils.text import (
    contains_non_latin,
    latest_supplier_body,
    normalise_text,
    split_sentences,
)


class SecurityFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule: str
    matched_text: str
    source_location: str
    confidence: float
    details: str | None = None


SAFE_PHRASES = (
    r"ai-powered",
    r"system integration team",
    r"erp system",
    r"assistant manager",
    r"no automatic renewal",
    r"renewal only if both parties sign again",
    r"not asking for exclusivity",
    r"no exclusivity",
    r"not asking for an exclusive",
)

ACTOR_RE = re.compile(
    r"(?i)\b(?:ai|a\.i\.|artificial intelligence|language model|llm|chatbot|bot|"
    r"automated system|automated agent|ai agent|agent|assistant|reviewer|reviewing assistant)\b"
)
DIRECTIVE_RE = re.compile(
    r"(?i)\b(?:accept|accepted|approve|approved|pre-?approved|override|ignore|"
    r"skip|bypass|mark|record|output|action\s*=|signed off|sign off|granted)\b"
)
EXPLICIT_PATTERNS = (
    ("system_note", re.compile(r"(?i)\bsystem\s+note\s+to\s+(?:the\s+)?(?:ai|agent|model|bot)")),
    ("note_language_model", re.compile(r"(?i)\bnote\s+for\s+the\s+language\s+model\b")),
    ("bot_screens", re.compile(r"(?i)\bwhichever\s+(?:bot|ai|assistant|automated\s+system)\b")),
    ("automated_reader", re.compile(r"(?i)\bif\s+an\s+automated\s+system\s+is\s+reading\b")),
    ("html_comment", re.compile(r"(?is)<!--.*?(?:assistant|ai|bot|agent).*?(?:accept|approve|override).*?-->")),
    ("admin_override", re.compile(r"(?i)\[?\s*admin\s+override\s*\]?")),
    ("ignore_instructions", re.compile(r"(?i)\bignore\s+(?:your|all|the)\s+previous\s+instructions\b")),
    ("policy_claim", re.compile(r"(?i)\b(?:pre-?approved|approval\s+already\s+granted|already\s+signed\s+off)\b")),
    ("skip_checks", re.compile(r"(?i)\bskip\s+the\s+playbook\s+checks\b")),
    ("accept_output", re.compile(r"(?i)\b(?:output|set\s+action\s*=)\s*accept\b")),
)


def _strip_safe_phrases(text: str) -> str:
    cleaned = text
    for pattern in SAFE_PHRASES:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def detect_prompt_injection(thread_message: str) -> list[SecurityFinding]:
    """Scan the entire thread, not only the latest message."""
    text = normalise_text(thread_message)
    findings: list[SecurityFinding] = []
    for rule_id, pattern in EXPLICIT_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                SecurityFinding(
                    rule=f"PROMPT_INJECTION:{rule_id}",
                    matched_text=match.group(0),
                    source_location=f"character {match.start()}",
                    confidence=0.99,
                    details="explicit instruction/override pattern",
                )
            )
    # Sentence-level semantic detection on text with known harmless phrases removed.
    cleaned = _strip_safe_phrases(text)
    for sentence in split_sentences(cleaned):
        actors = list(ACTOR_RE.finditer(sentence))
        if not actors:
            continue
        directives = list(DIRECTIVE_RE.finditer(sentence))
        if not directives:
            continue
        # Require a directive in the same sentence and a plausible target. This keeps
        # ordinary product descriptions from being classified as injections.
        matched = sentence.strip()
        findings.append(
            SecurityFinding(
                rule="PROMPT_INJECTION:ai_directed_instruction",
                matched_text=matched,
                source_location="thread sentence",
                confidence=0.93,
                details="AI/automation/reviewer target plus an action directive",
            )
        )
    if contains_non_latin(text):
        # Multilingual injection: a non-Latin script sentence that also contains an
        # explicit approval marker is treated as a directive aimed at the reader.
        for sentence in split_sentences(text):
            if contains_non_latin(sentence) and re.search(
                r"(?i)\b(?:accept|approve|approved|override|instruction)\b", sentence
            ):
                findings.append(
                    SecurityFinding(
                        rule="PROMPT_INJECTION:multilingual_directive",
                        matched_text=sentence.strip(),
                        source_location="thread sentence",
                        confidence=0.9,
                        details="non-Latin directive containing an approval marker",
                    )
                )
    # Deduplicate identical findings while preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[SecurityFinding] = []
    for finding in findings:
        key = (finding.rule, finding.matched_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def _percentage_in_sentence(sentence: str) -> Decimal | None:
    match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*%", sentence)
    if not match:
        return None
    return Decimal(match.group(1))


def _contains_advance_language(sentence: str) -> bool:
    return bool(
        re.search(
            r"(?i)\b(?:advance|upfront|up-front|prepay|pre-payment|deposit|in advance|order value)\b",
            sentence,
        )
    )


def _fraction_percentage(sentence: str) -> Decimal | None:
    lower = sentence.lower()
    if re.search(r"(?i)\bhalf\b", lower):
        return Decimal(50)
    if re.search(r"(?i)\bone[-\s]?third\b", lower):
        return Decimal(100) / Decimal(3)
    if re.search(r"(?i)\btwo[-\s]?thirds\b", lower):
        return Decimal(200) / Decimal(3)
    if re.search(r"(?i)\bquarter\b", lower):
        return Decimal(25)
    return None


def detect_banned_terms(latest_message: str) -> list[SecurityFinding]:
    text = normalise_text(latest_message)
    findings: list[SecurityFinding] = []

    def add(rule: str, sentence: str, details: str) -> None:
        findings.append(
            SecurityFinding(
                rule=f"BANNED_TERM:{rule}",
                matched_text=sentence.strip(),
                source_location="latest supplier message",
                confidence=0.96,
                details=details,
            )
        )

    for sentence in split_sentences(text):
        # Automatic renewal, excluding the explicitly allowed negative/mutual wording.
        safe_renewal = bool(
            re.search(r"(?i)\bno\s+automatic\s+renewal\b", sentence)
            or re.search(r"(?i)\brenewal\s+only\s+(?:if|by)\b", sentence)
            or re.search(r"(?i)\brenewal\s+by\s+mutual\s+agreement\b", sentence)
        )
        if not safe_renewal:
            renewal_patterns = (
                r"\bauto[-\s]?renew(?:s|ed|ing|al)?\b",
                r"\brenews?\s+(?:on\s+its\s+own|automatically)\b",
                r"\bautomatically\s+renews?\b",
                r"\bcontinues?\s+(?:from\s+)?year\s+to\s+year\b",
                r"\brenews?\s+every\s+year\s+unless\b",
            )
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in renewal_patterns):
                add("AUTOMATIC_RENEWAL", sentence, "automatic renewal language")

        # Advance/upfront payment above 30%.
        if _contains_advance_language(sentence):
            percent = _percentage_in_sentence(sentence)
            if percent is None:
                percent = _fraction_percentage(sentence)
            if percent is not None and percent > Decimal(30):
                add(
                    "ADVANCE_PAYMENT_ABOVE_30_PERCENT",
                    sentence,
                    f"advance/upfront payment of {percent}% exceeds the allowed 30%",
                )

        # Unlimited liability for Acme.
        if re.search(r"(?i)\bliabilit\w*\b[^.]{0,100}\bunlimited\b", sentence) or re.search(
            r"(?i)\bunlimited\b[^.]{0,100}\bliabilit", sentence
        ):
            add("UNLIMITED_LIABILITY", sentence, "unlimited liability language")

        # Exclusivity, excluding explicit no-exclusivity statements.
        safe_exclusivity = bool(
            re.search(
                r"(?i)\b(?:not\s+asking\s+for|no|without|does\s+not\s+require|not\s+seeking)\s+(?:an?\s+)?exclusiv",
                sentence,
            )
        )
        if not safe_exclusivity and re.search(
            r"(?i)\bexclusive\s+supplier\b|\bsole\s+supplier\b|\bbuy\b[^.]{0,80}\bonly\s+from\s+us\b|\bexclusivity\b|\bexclusive\b",
            sentence,
        ):
            # The playbook allows 'no exclusivity is asked'; the negation check above
            # handles those statements. Affirmative exclusivity is banned.
            if not re.search(r"(?i)\bnot\s+exclusive\b|\bno\s+exclusiv", sentence):
                add("EXCLUSIVITY", sentence, "exclusivity obligation")
    deduped: list[SecurityFinding] = []
    seen: set[tuple[str, str]] = set()
    for finding in findings:
        key = (finding.rule, finding.matched_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def scan_security(thread_message: str) -> tuple[list[SecurityFinding], list[SecurityFinding]]:
    """Return (injection findings, banned-term findings in latest supplier message)."""
    injections = detect_prompt_injection(thread_message)
    banned = detect_banned_terms(latest_supplier_body(thread_message))
    return injections, banned
