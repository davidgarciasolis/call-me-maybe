"""Enmascarado de logits comparando el texto decodificado por el SDK."""

from math import isfinite
from typing import Any

from .restricciones import estado_etapas, preparar_etapas


def enmascarar_logits(
    logits: list[float],
    modelo: Any,
    ids_generados: list[int],
    tokens_permitidos: set[int],
    funciones: list[dict[str, Any]],
) -> int:
    """Bloquea los tokens cuyo texto incumple las restricciones.

    Decodifica cada candidato junto con la respuesta generada. Rechaza
    texto vacío, caracteres incompletos representados por '�' y candidatos
    que cambien el texto anterior. Modifica los logits antes de seleccionar.
    """
    identificador: int
    logit: float
    respuesta: str
    texto: str
    permitido: bool
    cantidad: int
    etapas: list[list[tuple[str, str]]]

    respuesta = modelo.decode(ids_generados)
    etapas = [preparar_etapas(funcion) for funcion in funciones]
    cantidad = 0

    for identificador, logit in enumerate(logits):
        permitido = False
        if isfinite(logit) and identificador in tokens_permitidos:
            texto = modelo.decode(ids_generados + [identificador])
            if (texto.startswith(respuesta) and len(texto) > len(respuesta)
                    and "�" not in texto):
                permitido = estado_etapas(texto, etapas) != "invalida"

        if permitido:
            cantidad += 1
        else:
            logits[identificador] = float("-inf")

    return cantidad
