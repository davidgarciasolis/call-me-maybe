"""Select tokens by logit, checking text decoded by the SDK."""

from typing import Any

import numpy as np

from .constraints import is_valid_call_prefix


def prepare_tokens(
    model: Any,
    vocabulary: dict[str, int],
) -> dict[int, str]:
    """Decode each vocabulary ID once."""
    tokens: dict[int, str]
    token_id: int

    tokens = {}
    for token_id in set(vocabulary.values()):
        tokens[token_id] = model.decode([token_id])
    return tokens


def select_token(
    logits: list[float],
    response: str,
    decoded_tokens: dict[int, str],
    functions: list[dict[str, Any]],
) -> int:
    """Find the maximum and discard invalid candidates with -inf.

    Keep the lowest ID when logits are tied.
    Validate the accumulated response plus each candidate from the start.
    Do not modify the logits or call the SDK.
    """
    values: list[float]
    token_id: int
    fragment: str

    values = logits.copy()

    while True:
        token_id = int(np.argmax(values))
        if values[token_id] == float("-inf"):
            break
        fragment = decoded_tokens[token_id]
        if is_valid_call_prefix(response + fragment, functions):
            return token_id
        values[token_id] = float("-inf")

    raise ValueError("No allowed tokens remain.")
