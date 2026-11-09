"""Selección por logit comprobando el texto decodificado por el SDK."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

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

    Solo admite logits finitos. En empates conserva el menor ID.
    Valida desde el principio la respuesta acumulada más cada candidato.
    No modifica los logits ni llama al SDK.
    """
    valores: NDArray[np.float64]
    identificador: int
    fragmento: str

    valores = np.array(logits, dtype=np.float64)
    valores[~np.isfinite(valores)] = -np.inf

    while valores.size > 0:
        identificador = int(np.argmax(valores))
        if valores[identificador] == -np.inf:
            break
        fragmento = tokens_decodificados.get(identificador, "")
        if fragmento and prefijo_llamada_valido(
            respuesta + fragmento, funciones,
        ):
            return identificador
        valores[identificador] = -np.inf

    raise ValueError("No quedan tokens permitidos.")
