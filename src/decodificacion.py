"""Enmascarado de logits comparando el texto decodificado por el SDK."""

from math import isfinite
from typing import Any

from .restricciones import Progreso, avanzar, avanzar_funciones


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
    progresos: list[Progreso | None],
    tokens_decodificados: dict[int, str],
    etapas_funciones: list[list[tuple[str, str]]],
) -> int:
    """Bloquea los tokens cuyo texto incumple las restricciones.

    Prueba el texto de cada token desde el progreso guardado por función.
    Reutiliza el avance del primer carácter entre candidatos de este paso.
    No modifica esos progresos ni llama al SDK; solo modifica los logits.
    """
    identificador: int
    logit: float
    fragmento: str
    permitido: bool
    cantidad: int
    etapas: list[tuple[str, str]]
    progreso: Progreso | None
    inicios: dict[str, list[Progreso | None]]
    primero: str
    resto: str

    cantidad = 0
    inicios = {}

    for identificador, logit in enumerate(logits):
        permitido = False
        fragmento = tokens_decodificados.get(identificador, "")
        if isfinite(logit) and fragmento:
            primero = fragmento[0]
            if primero not in inicios:
                inicios[primero] = avanzar_funciones(
                    progresos, etapas_funciones, primero
                )
            resto = fragmento[1:]
            for progreso, etapas in zip(inicios[primero], etapas_funciones):
                if (progreso is not None
                        and avanzar(progreso, etapas, resto) is not None):
                    permitido = True
                    break

        if permitido:
            cantidad += 1
        else:
            logits[identificador] = float("-inf")

    return cantidad
