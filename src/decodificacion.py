"""Selección por logit comprobando el texto decodificado por el SDK."""

from math import isfinite
from typing import Any

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
    """Devuelve el ID del primer token válido por logit descendente.

    Solo ordena logits finitos. En empates conserva el menor ID.
    Valida desde el principio la respuesta acumulada más cada candidato.
    No modifica los logits ni llama al SDK.
    """
    identificadores: list[int]
    identificador: int
    logit: float
    fragmento: str

    identificadores = []
    for identificador, logit in enumerate(logits):
        if isfinite(logit):
            identificadores.append(identificador)
    identificadores.sort(key=logits.__getitem__, reverse=True)

    for identificador in identificadores:
        fragmento = tokens_decodificados.get(identificador, "")
        if not fragmento:
            continue
        if prefijo_llamada_valido(respuesta + fragmento, funciones):
            return identificador

    raise ValueError("No quedan tokens permitidos.")
