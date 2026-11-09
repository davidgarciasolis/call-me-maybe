"""Selección por logit comprobando el texto decodificado por el SDK."""

from typing import Any

import numpy as np

from .restricciones import prefijo_llamada_valido


def preparar_tokens(
    modelo: Any,
    vocabulario: dict[str, int],
) -> dict[int, str]:
    """Decodifica una vez cada ID y descarta texto vacío o con '�'."""
    tokens: dict[int, str]
    identificador: int
    texto: str

    tokens = {}
    for identificador in set(vocabulario.values()):
        texto = modelo.decode([identificador])
        if texto and "�" not in texto:
            tokens[identificador] = texto
    return tokens


def seleccionar_token(
    logits: list[float],
    respuesta: str,
    tokens_decodificados: dict[int, str],
    funciones: list[dict[str, Any]],
) -> int:
    """Busca el máximo y descarta con -inf los candidatos inválidos.

    En empates conserva el menor ID.
    Valida desde el principio la respuesta acumulada más cada candidato.
    No modifica los logits ni llama al SDK.
    """
    valores: list[float]
    identificador: int
    fragmento: str

    valores = logits.copy()

    while valores:
        identificador = int(np.argmax(valores))
        if valores[identificador] == float("-inf"):
            break
        fragmento = tokens_decodificados[identificador]
        if prefijo_llamada_valido(respuesta + fragmento, funciones):
            return identificador
        valores[identificador] = float("-inf")

    raise ValueError("No quedan tokens permitidos.")
