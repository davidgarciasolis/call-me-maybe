"""Enmascarado de logits reutilizando los bytes del vocabulario."""

from codecs import utf_8_decode
from math import isfinite
import re
from typing import Any

from .restricciones import cadena_abierta, estado_etapas, preparar_etapas


def preparar_tokens(
    vocabulario: dict[str, int],
    tokenizador: dict[str, Any],
) -> dict[int, bytes]:
    """Prepara una vez los bytes ByteLevel y excluye tokens añadidos.

    Rechaza decoders desconocidos para evitar interpretar mal el vocabulario.
    """
    modelo: Any
    anadidos: Any
    anadido: Any
    id_anadido: Any
    excluidos: set[int]
    tokens: dict[int, bytes]
    decoder: Any
    valores: list[int]
    mapa: dict[str, int]
    valor: int
    siguiente: int
    token: str
    identificador: int

    modelo = tokenizador.get("model")
    if not isinstance(modelo, dict) or modelo.get("vocab") != vocabulario:
        raise ValueError("El vocabulario no coincide con tokenizer.json.")

    decoder = tokenizador.get("decoder")
    if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
        raise ValueError("La máscara optimizada requiere decoder ByteLevel.")

    # ByteLevel representa los 256 bytes mediante caracteres imprimibles.
    valores = list(range(33, 127)) + list(range(161, 173))
    valores += list(range(174, 256))
    mapa = {chr(valor): valor for valor in valores}
    siguiente = 256
    for valor in range(256):
        if valor not in valores:
            mapa[chr(siguiente)] = valor
            siguiente += 1

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

    tokens = {}
    for token, identificador in vocabulario.items():
        if token and identificador not in excluidos:
            try:
                tokens[identificador] = bytes(mapa[letra] for letra in token)
            except KeyError as error:
                raise ValueError(
                    f"Token ByteLevel no válido: {token!r}."
                ) from error

    return tokens


def enmascarar_logits(
    logits: list[float],
    ids_generados: list[int],
    tokens_permitidos: dict[int, bytes],
    funciones: list[dict[str, Any]],
) -> int:
    """Bloquea candidatos inválidos sin decodificarlos mediante el SDK.

    Reutiliza las etapas y la validez del primer carácter en cada paso.
    Conserva UTF-8 incompleto solo donde cabe contenido de una cadena.
    Modifica todos los logits inválidos antes de seleccionar el siguiente.
    """
    identificador: int
    logit: float
    permitido: bool
    cantidad: int
    datos: bytes
    pendientes: bytes
    fragmento: bytes
    respuesta: str
    texto: str
    consumidos: int
    etapas: list[list[tuple[str, str]]]
    inicios: dict[str, bool]
    primero: str
    cadena_libre: bool
    texto_normal: re.Pattern[str]

    datos = b"".join(tokens_permitidos[token] for token in ids_generados)
    respuesta, consumidos = utf_8_decode(datos, "strict", False)
    pendientes = datos[consumidos:]
    etapas = [preparar_etapas(funcion) for funcion in funciones]
    cadena_libre = cadena_abierta(respuesta, etapas)
    texto_normal = re.compile(r'[^"\\\x00-\x1f]+')
    inicios = {}
    cantidad = 0
    for identificador, logit in enumerate(logits):
        permitido = False
        fragmento = tokens_permitidos.get(identificador, b"")
        if isfinite(logit) and fragmento:
            try:
                datos = pendientes + fragmento
                texto, consumidos = utf_8_decode(datos, "strict", False)
                if consumidos < len(datos):
                    # Un carácter de prueba exige estar dentro de una cadena
                    # libre, fuera de escapes y del texto fijo de la llamada.
                    texto += "¡"
                if cadena_libre and texto_normal.fullmatch(texto):
                    permitido = True
                elif texto:
                    primero = texto[0]
                    if primero not in inicios:
                        inicios[primero] = estado_etapas(
                            respuesta + primero, etapas
                        ) != "invalida"
                    if inicios[primero]:
                        permitido = estado_etapas(
                            respuesta + texto, etapas
                        ) != "invalida"
            except UnicodeDecodeError:
                permitido = False

        if permitido:
            cantidad += 1
        else:
            logits[identificador] = float("-inf")

    return cantidad
