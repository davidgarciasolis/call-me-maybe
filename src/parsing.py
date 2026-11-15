"""Read and validate input files."""

import json
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ValidationError, field_validator


class Parameter(BaseModel):
    """Define a parameter with one of the supported types."""

    type: Literal["string", "number", "boolean"]


class Return(BaseModel):
    """Define the return type of a function."""

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Reject an empty or whitespace-only type."""
        if not value.strip():
            raise ValueError("The return type cannot be empty.")
        return value


class Function(BaseModel):
    """Define the structure of a function available to the model."""

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Return

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject an empty or whitespace-only name."""
        if not value.strip():
            raise ValueError("The name cannot be empty.")
        return value


class Request(BaseModel):
    """Define a natural language request containing a prompt field."""

    prompt: str


def read_json(path: str) -> Any:
    """Read a JSON file and return its contents as Python data."""
    file: TextIO

    with open(path) as file:
        return json.load(file)


def load_functions(path: str) -> list[dict[str, Any]]:
    """Read and validate functions, preserving the original dictionaries."""
    functions: Any
    names: set[str]
    index: int
    function: Any
    context: str
    validated: Function

    functions = read_json(path)
    if not isinstance(functions, list):
        raise ValueError(f"{path}: expected a list of functions.")

    names = set()
    for index, function in enumerate(functions, start=1):
        context = f"{path}: function {index}"
        try:
            validated = Function.model_validate(function, strict=True)
        except ValidationError as error:
            raise ValueError(f"{context}: {error}") from error

        if validated.name in names:
            raise ValueError(
                f"{context}: duplicate name '{validated.name}'."
            )
        names.add(validated.name)

    return functions


def load_prompts(path: str) -> list[dict[str, Any]]:
    """Read and validate requests, preserving the original dictionaries."""
    prompts: Any
    index: int
    prompt: Any
    context: str

    prompts = read_json(path)
    if not isinstance(prompts, list):
        raise ValueError(f"{path}: expected a list of prompts.")

    for index, prompt in enumerate(prompts, start=1):
        context = f"{path}: prompt {index}"
        try:
            Request.model_validate(prompt, strict=True)
        except ValidationError as error:
            raise ValueError(f"{context}: {error}") from error

    return prompts
