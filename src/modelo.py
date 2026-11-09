"""Inicialización del modelo y carga de su vocabulario."""

from llm_sdk import Small_LLM_Model

from .error import ErrorCargaModelo, ErrorCargaVocabulario
from .parseo import read_json


def inicializar_modelo() -> Small_LLM_Model:
    """Carga el modelo Qwen y su tokenizador."""
    modelo: Small_LLM_Model

    try:
        modelo = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    except Exception as error:
        raise ErrorCargaModelo(str(error)) from error
    return modelo


def cargar_vocabulario(modelo: Small_LLM_Model) -> dict[str, int]:
    """Lee las correspondencias entre tokens e identificadores del modelo."""
    ruta: str
    vocabulario: dict[str, int]

    try:
        ruta = modelo.get_path_to_vocab_file()
        vocabulario = read_json(ruta)
    except Exception as error:
        raise ErrorCargaVocabulario(str(error)) from error

    return vocabulario
