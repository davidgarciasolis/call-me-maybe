"""Enmascarado de logits mediante el texto devuelto por el SDK."""

from math import isfinite
from typing import Any

from .restricciones import estado_llamada


def preparar_tokens(
    vocabulario: dict[str, int],
    tokenizador: dict[str, Any],
) -> set[int]:
    """Obtiene los IDs ordinarios, excluyendo tokens añadidos y vacíos.

    Recibe el vocabulario validado y el contenido de tokenizer.json.
    Lanza ValueError si el vocabulario no coincide o si los datos de los
    tokens añadidos son incorrectos. La decodificación se delega al SDK.
    """
    modelo: Any
    anadidos: Any
    anadido: Any
    id_anadido: Any
    excluidos: set[int]
    tokens: set[int]
    token: str
    identificador: int

    modelo = tokenizador.get("model")
    if not isinstance(modelo, dict) or modelo.get("vocab") != vocabulario:
        raise ValueError("El vocabulario no coincide con tokenizer.json.")

    anadidos = tokenizador.get("added_tokens", [])
    if not isinstance(anadidos, list):
        raise ValueError("'added_tokens' debe ser una lista.")

    excluidos = set()
    for anadido in anadidos:
        if not isinstance(anadido, dict):
            raise ValueError("Cada token añadido debe ser un objeto con ID.")
        id_anadido = anadido.get("id")
        if type(id_anadido) is not int or id_anadido < 0:
            raise ValueError("Cada token añadido necesita un ID no negativo.")
        excluidos.add(id_anadido)

    tokens = set()
    for token, identificador in vocabulario.items():
        if token and identificador not in excluidos:
            tokens.add(identificador)

    return tokens


def candidato_valido(
    modelo: Any,
    ids_generados: list[int],
    identificador: int,
    funciones: list[dict[str, Any]],
) -> bool:
    """Valida el texto de la respuesta completa con el token candidato.

    Recibe solo IDs de la respuesta, sin los del prompt. Decodifica juntos
    todos los tokens para respetar los caracteres divididos entre ellos.
    No modifica los IDs recibidos. La validación se realiza sobre el texto
    del SDK, que puede contener caracteres de reemplazo U+FFFD.
    """
    candidata: str

    candidata = modelo.decode(ids_generados + [identificador])
    if estado_llamada(candidata, funciones) == "invalida":
        return False
    return True


def enmascarar_logits(
    logits: list[float],
    modelo: Any,
    ids_generados: list[int],
    tokens_permitidos: set[int],
    funciones: list[dict[str, Any]],
) -> int:
    """Pone en -inf los logits inválidos y cuenta los permitidos.

    Bloquea IDs excluidos, logits no finitos y textos que incumplen la
    gramática. Recibe funciones validadas con los tipos de estado_llamada().
    Modifica logits, pero conserva ids_generados y tokens_permitidos.
    Si devuelve cero, el llamador debe gestionar el error antes de elegir.
    Los errores del SDK se propagan al llamador.
    """
    identificador: int
    logit: float
    permitido: bool
    cantidad: int

    cantidad = 0
    for identificador, logit in enumerate(logits):
        permitido = False
        if isfinite(logit) and identificador in tokens_permitidos:
            permitido = candidato_valido(
                modelo, ids_generados, identificador, funciones
            )

        if permitido:
            cantidad += 1
        else:
            logits[identificador] = float("-inf")

    return cantidad
