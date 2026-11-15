"""Write the JSON file containing generated calls."""

import json
from pathlib import Path
from typing import Any, TextIO


def save_results(
    path: str,
    results: list[dict[str, Any]],
) -> None:
    """Save the JSON array, creating any necessary directories.

    Overwrite the file if it exists. Propagate access and serialization
    errors so the caller can report the cause of the failure.
    """
    destination: Path
    file: TextIO

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with open(destination, "w") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4,
            allow_nan=False,
        )
