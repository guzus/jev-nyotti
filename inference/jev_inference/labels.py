from __future__ import annotations

import itertools
import string
from dataclasses import dataclass
from typing import Protocol

from .settings import MAX_OPTIONS


class LabelTokenizer(Protocol):
    all_special_ids: list[int]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int], **kwargs) -> str: ...


@dataclass(frozen=True)
class Label:
    text: str
    token_id: int


def select_labels(tokenizer: LabelTokenizer, assistant_prefix: str, count: int = MAX_OPTIONS) -> list[Label]:
    """Select unique single-token ASCII labels and verify the continuation boundary.

    Numbers above 9 and multi-letter combinations are not assumed to be single
    tokens. Every label is checked against the pinned tokenizer. Short letters
    come first for common 2-5 option classifications.
    """
    if count < 1 or count > MAX_OPTIONS:
        raise ValueError("unsupported label count")
    prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)
    special_ids = set(tokenizer.all_special_ids)
    seen_ids: set[int] = set()
    labels: list[Label] = []
    alphabet = string.ascii_uppercase + string.ascii_lowercase
    candidates = itertools.chain(
        alphabet,
        ("".join(pair) for pair in itertools.product(string.ascii_uppercase, repeat=2)),
        ("".join(pair) for pair in itertools.product(string.ascii_lowercase, repeat=2)),
    )
    for text in candidates:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1 or token_ids[0] in special_ids or token_ids[0] in seen_ids:
            continue
        if tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) != text:
            continue
        # Ensures the label is one *next* token after the actual chat-template suffix.
        if tokenizer.encode(assistant_prefix + text, add_special_tokens=False) != prefix_ids + token_ids:
            continue
        labels.append(Label(text=text, token_id=token_ids[0]))
        seen_ids.add(token_ids[0])
        if len(labels) == count:
            return labels
    raise RuntimeError(f"tokenizer supplied fewer than {count} usable single-token labels")
