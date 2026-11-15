"""Validate the entire response from the beginning, without stages."""

import json
from typing import Any

from .validation import is_valid_value_prefix, is_complete_value


def consume_fixed(
    response: str, position: int, expected: str,
) -> int | None:
    """Return the expected position, or None if the text does not match.

    If it exceeds the response length, the fixed text is incomplete.
    """
    fragment: str

    fragment = response[position:position + len(expected)]
    if expected.startswith(fragment):
        return position + len(expected)
    return None


def consume_string(response: str, position: int) -> int | None:
    """Scan a string; return the available end, or None if invalid.

    Allow a pending closing quote or escape at the end of available text.
    Reject empty strings, but allow spaces as content.
    Preserve the simple escapes supported by the previous version.
    """
    character: str

    if response[position] != '"':
        return None
    if response[position:position + 2] == '""':
        return None
    position += 1
    while position < len(response):
        character = response[position]
        position += 1
        if character == '"':
            return position
        elif ord(character) < 32:
            return None
    return position


def is_valid_function_prefix(
    response: str, function: dict[str, Any],
) -> bool:
    """Check the entire response against a function, without saving progress.

    Require compact JSON, ordered parameters and string/number/boolean types.
    Accept incomplete prefixes that can still form the call.
    """
    position: int | None
    start: int
    name: str
    definition: dict[str, Any]
    value_type: str
    value: str
    separator: str

    position = consume_fixed(
        response, 0,
        '{"name":' + json.dumps(function["name"]) + ',"parameters":{',
    )
    if position is None:
        return False
    if position >= len(response):
        return True

    separator = ""
    for name, definition in function["parameters"].items():
        position = consume_fixed(
            response, position, separator + json.dumps(name) + ":",
        )
        if position is None:
            return False
        if position >= len(response):
            return True

        start = position
        value_type = definition["type"]
        if value_type == "string":
            position = consume_string(response, position)
            if position is None:
                return False
            if position == len(response):
                return True
        else:
            while (position < len(response)
                   and response[position] not in ",}"):
                position += 1
            value = response[start:position]
            if position == len(response):
                return is_valid_value_prefix(value, value_type)
            if not is_complete_value(value, value_type):
                return False
        separator = ","

    position = consume_fixed(response, position, "}}")
    if position is None:
        return False
    return position >= len(response)


def is_valid_call_prefix(
    response: str, functions: list[dict[str, Any]],
) -> bool:
    """Accept a response that can complete a call to an available function."""
    function: dict[str, Any]

    for function in functions:
        if is_valid_function_prefix(response, function):
            return True
    return False
