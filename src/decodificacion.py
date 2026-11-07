"""Enmascarado de logits comparando el texto decodificado por el SDK."""

from math import isfinite
from typing import Any

from .restricciones import estado_etapas, preparar_etapas


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


def enmascarar_logits(
    logits: list[float],
    respuesta: str,
    tokens_decodificados: dict[int, str],
    funciones: list[dict[str, Any]],
) -> int:
    """Bloquea los tokens cuyo texto incumple las restricciones.

    Concatena la respuesta con el texto guardado de cada token, sin llamar
    al SDK. Modifica los logits antes de seleccionar el siguiente token.
    """
    identificador: int
    logit: float
    fragmento: str
    texto: str
    permitido: bool
    cantidad: int
    etapas: list[list[tuple[str, str]]]

    etapas = [preparar_etapas(funcion) for funcion in funciones]
    cantidad = 0

    for identificador, logit in enumerate(logits):
        permitido = False
        fragmento = tokens_decodificados.get(identificador, "")
        if isfinite(logit) and fragmento:
            texto = respuesta + fragmento
            permitido = estado_etapas(texto, etapas) != "invalida"

        if permitido:
            cantidad += 1
        else:
            logits[identificador] = float("-inf")

    return cantidad
