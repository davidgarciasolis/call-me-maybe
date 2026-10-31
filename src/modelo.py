"""Inicialización del modelo y carga de su vocabulario."""

import os
from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model

from .error import ErrorCargaModelo, ErrorCargaVocabulario
from .parseo import read_json


def inicializar_modelo(
    nombre: str = "Qwen/Qwen3-0.6B",
) -> Small_LLM_Model:
    """Carga el modelo elegido y reutiliza el login de Hugging Face."""
    modelo: Small_LLM_Model
    cache: Path
    cache_modelos: Path
    token: Path

    try:
        cache = Path(os.environ.get(
            "XDG_CACHE_HOME", str(Path.home() / ".cache"),
        )) / "huggingface"
        cache_modelos = Path(os.environ.get("HF_HOME", str(cache)))
        token = cache / "token"
        if (
            "HF_TOKEN" not in os.environ
            and "HF_TOKEN_PATH" not in os.environ
            and token.is_file()
            and not (cache_modelos / "token").is_file()
        ):
            os.environ.setdefault("HF_TOKEN", token.read_text().strip())
        modelo = Small_LLM_Model(model_name=nombre)
    except Exception as error:
        raise ErrorCargaModelo(str(error)) from error
    return modelo


def cargar_vocabulario(modelo: Small_LLM_Model) -> dict[str, int]:
    """Lee y valida las correspondencias entre tokens e identificadores."""
    tokenizador: Any
    vocabulario: Any
    token: str
    identificador: Any

    try:
        tokenizador = read_json(modelo.get_path_to_tokenizer_file())
        vocabulario = tokenizador["model"]["vocab"]

        if not isinstance(vocabulario, dict) or not vocabulario:
            raise ValueError(
                "El vocabulario debe ser un objeto JSON no vacío."
            )

        for token, identificador in vocabulario.items():
            if type(identificador) is not int or identificador < 0:
                raise ValueError(f"ID no válido para el token '{token}'.")
    except Exception as error:
        raise ErrorCargaVocabulario(str(error)) from error

    return vocabulario
