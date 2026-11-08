"""Selección por logit comprobando el texto decodificado por el SDK."""

from math import isfinite
from typing import Any

from .restricciones import Progreso, avanzar


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
    progresos: list[Progreso | None],
    tokens_decodificados: dict[int, str],
    etapas_funciones: list[list[tuple[str, str]]],
) -> int:
    """Devuelve el ID del primer token válido por logit descendente.

    Solo ordena logits finitos. En empates conserva el menor ID.
    No modifica los logits ni los progresos, ni llama al SDK.
    """
    identificadores: list[int]
    identificador: int
    logit: float
    fragmento: str
    etapas: list[tuple[str, str]]
    progreso: Progreso | None

    identificadores = []
    for identificador, logit in enumerate(logits):
        if isfinite(logit):
            identificadores.append(identificador)
    identificadores.sort(key=logits.__getitem__, reverse=True)

    for identificador in identificadores:
        fragmento = tokens_decodificados.get(identificador, "")
        if not fragmento:
            continue
        for progreso, etapas in zip(progresos, etapas_funciones):
            if (progreso is not None
                    and avanzar(progreso, etapas, fragmento) is not None):
                return identificador

    raise ValueError("No quedan tokens permitidos.")
