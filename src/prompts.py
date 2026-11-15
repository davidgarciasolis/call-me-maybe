"""Build instructions for the model."""

import json
from typing import Any


def build_prompt(
    functions: list[dict[str, Any]],
    prompt: str,
) -> str:
    """Combine the instructions, functions and request."""
    instructions: str
    functions_json: str
    text: str

    instructions = (
        "Choose one of the available functions for the user's request.\n"
        "Extract its arguments from the request. Do not execute it.\n"
        "Return only a JSON object with the keys "
        '"name" and "parameters", in that order.\n'
        "Write compact JSON on a single line, without spaces outside "
        "string values or leading or trailing whitespace.\n"
        "Escape special characters as required by JSON. "
        "Do not add extra keys, explanations or Markdown.\n"
        "Use the exact function name and parameter names, "
        "and respect the declared parameter types.\n"
    )

    functions_json = json.dumps(functions)

    text = (
        f"{instructions}\n"
        f"Available functions:\n{functions_json}\n\n"
        f"User request:\n{prompt}\n\n"
        "Response:\n"
    )

    return text
