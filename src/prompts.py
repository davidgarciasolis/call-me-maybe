"""Construcción de las instrucciones para el modelo."""

import json
from typing import Any


def construir_prompt(
    funciones: list[dict[str, Any]],
    promt: str,
) -> str:
    """Combina las instrucciones, las funciones y la petición."""
    instrucciones: str
    funciones_json: str
    texto: str

    instrucciones = (
        "Choose one of the available functions for the user's request.\n"
        "Extract its arguments from the request. Do not execute it.\n"
        "Return only a JSON object with the keys "
        '"name" and "parameters", in that order.\n'
        "Escape special characters as required by JSON. "
        "Do not add extra keys, explanations or Markdown.\n"
        "Use the exact function name and parameter names, "
        "and respect the declared parameter types.\n"
    )

    funciones_json = json.dumps(funciones)

    texto = (
        f"{instrucciones}\n"
        f"Available functions:\n{funciones_json}\n\n"
        f"User request:\n{promt}\n\n"
        "Response:\n"
    )

    return texto
