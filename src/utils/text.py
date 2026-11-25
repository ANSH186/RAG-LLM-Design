"""Text normalisation, thread segmentation and tokenisation helpers."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

BLOCK_RE = re.compile(
    r"^\[(Earlier|Latest) message — (?P<speaker>[^,\]]+?)(?:, (?P<date>\d{4}-\d{2}-\d{2}))?\]\s*$",
    re.MULTILINE,
)

ACME_SPEAKER = "acme procurement"


@dataclass(frozen=True)
class MessageBlock:
    kind: str
    speaker: str
    date: str | None
    text: str


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("—", "-").replace("–", "-")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return text


def split_message_blocks(message: str) -> list[MessageBlock]:
    """Split a CSV supplier_message into labelled blocks.

    A message without block markers is a single supplier block.
    """
    matches = list(BLOCK_RE.finditer(message))
    if not matches:
        return [MessageBlock("latest", "Supplier", None, message.strip())]
    blocks: list[MessageBlock] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(message)
        kind = match.group(1).lower()
        speaker = match.group("speaker").strip()
        blocks.append(MessageBlock(kind, speaker, match.group("date"), message[start:end].strip()))
    return blocks


def is_acme(speaker: str) -> bool:
    return speaker.strip().lower() == ACME_SPEAKER


def latest_supplier_block(message: str) -> MessageBlock:
    blocks = split_message_blocks(message)
    for block in reversed(blocks):
        if not is_acme(block.speaker):
            return block
    return MessageBlock("latest", "Supplier", None, message.strip())


def previous_supplier_blocks(message: str) -> list[MessageBlock]:
    blocks = split_message_blocks(message)
    return [block for block in blocks if not is_acme(block.speaker)]


def latest_supplier_body(message: str) -> str:
    return latest_supplier_block(message).text


def split_sentences(text: str) -> list[str]:
    """Conservative sentence/line splitter.

    Newlines are respected because generated supplier messages place the offer on
    its own line. Semicolons are not split: they often carry a second banned clause.
    """
    pieces: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on sentence-ending punctuation followed by whitespace.
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", line):
            part = part.strip()
            if part:
                pieces.append(part)
    return pieces


def tokenize(text: str) -> list[str]:
    """Stable lexical tokenizer used by BM25.

    Keeps numbers, currency markers, percentages and hyphenated terms. Normalises
    rupee spellings so 'Rs. 1200' and 'INR 1,200' share tokens.
    """
    lowered = normalise_text(text).lower()
    lowered = re.sub(r"\brs\.?\b", "inr", lowered)
    lowered = lowered.replace("₹", " inr ")
    lowered = re.sub(r"\b(?:rupees?|rupaye)\b", "inr", lowered)
    tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*%?", lowered)
    return tokens


def normalise_slug(value: str) -> str:
    value = normalise_text(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value


def contains_non_latin(text: str) -> bool:
    """True if a message contains letters outside the Latin script.

    Used only as one signal for multilingual prompt-injection detection, never to
    reject legitimate supplier prose on its own.
    """
    for char in text:
        if char.isalpha() and "LATIN" not in unicodedata.name(char, ""):
            return True
    return False


def excerpt(text: str, max_chars: int = 280) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "…"
