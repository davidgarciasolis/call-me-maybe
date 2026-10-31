"""Construcción de las instrucciones para el modelo."""

import json
from typing import Any


def construir_prompt(
    funciones: list[dict[str, Any]],
    promt: str,
    nombre_modelo: str = "Qwen/Qwen3-0.6B",
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
        "Write compact JSON on a single line, without spaces outside "
        "string values or leading or trailing whitespace.\n"
        "Escape special characters as required by JSON. "
        "Do not add extra keys, explanations or Markdown.\n"
        "Use the exact function name and parameter names, "
        "and respect the declared parameter types.\n"
        "Write every parameter in the order shown in its definition.\n"
        "Copy input strings as arguments; do not reverse, greet, "
        "calculate or otherwise execute the requested operation.\n"
        "For regex replacement, provide a regex pattern, not its result.\n"
    )

    funciones_json = json.dumps(funciones, separators=(",", ":"))

    if nombre_modelo == "LiquidAI/LFM2.5-230M":
        texto = (
            f"<|startoftext|><|im_start|>system\n{instrucciones}\n"
            f"List of tools: {funciones_json}<|im_end|>\n"
            f"<|im_start|>user\n{promt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return texto

    if nombre_modelo == "openai-community/gpt2":
        texto = (
            f"{instrucciones}\nAvailable functions:\n{funciones_json}\n"
            f"User request: {promt}\nFunction call:\n"
        )
        return texto

    if nombre_modelo == "google/gemma-3-270m-it":
        texto = (
            f"<bos><start_of_turn>user\n{instrucciones}\n"
            f"Available functions:\n{funciones_json}\n"
            f"User request:\n{promt}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        return texto

    texto = (
        f"<|im_start|>system\n{instrucciones}\n"
        f"Available functions:\n{funciones_json}<|im_end|>\n"
        f"<|im_start|>user\n{promt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    return texto
