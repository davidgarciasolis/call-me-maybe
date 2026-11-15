"""Validate JSON prefixes and values during generation."""

import json


def is_valid_number_prefix(fragment: str) -> bool:
    """Check a JSON number prefix, without exponents or whitespace.

    Allow integers and decimals. Empty text, a lone minus sign and a
    number ending in a decimal point are prefixes that can be completed.
    """
    parts: list[str]
    integer: str
    character: str

    if fragment.startswith("-"):
        fragment = fragment[1:]
    if not fragment:
        return True

    for character in fragment:
        if character not in "0123456789.":
            return False

    parts = fragment.split(".")
    if len(parts) > 2:
        return False

    integer = parts[0]
    if not integer:
        return False
    if len(integer) > 1 and integer[0] == "0":
        return False

    return True


def is_valid_boolean_prefix(fragment: str) -> bool:
    """Check if a fragment can form true or false, including empty text."""
    return "true".startswith(fragment) or "false".startswith(fragment)


def is_valid_value_prefix(fragment: str, value_type: str) -> bool:
    """Check a number or boolean prefix against the declared type."""
    if value_type == "number":
        return is_valid_number_prefix(fragment)

    if value_type == "boolean":
        return is_valid_boolean_prefix(fragment)

    return False


def is_complete_value(fragment: str, value_type: str) -> bool:
    """Check whether a valid prefix is already a complete JSON value."""
    if not is_valid_value_prefix(fragment, value_type):
        return False

    try:
        json.loads(fragment)
    except ValueError:
        return False

    return True
