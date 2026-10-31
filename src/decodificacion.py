"""Preparación de tokens para la decodificación restringida."""

import codecs
import re
from functools import lru_cache
from math import isfinite
from typing import Any

from .parseo import read_json
from .restricciones import (
    Restricciones,
    estado_restricciones,
    revisar_etapas,
    valor_completo,
)

ESPECIALES = re.compile(r'["\\\x00-\x1f]')


def preparar_tokens(
    modelo: Any,
    vocabulario: dict[str, int],
) -> dict[int, bytes]:
    """Relaciona los IDs del vocabulario con sus bytes originales.

    Lee tokenizer.json mediante el método público del SDK. Admite ByteLevel
    y la secuencia Replace/ByteFallback/Fuse de Gemma. Excluye los tokens
    añadidos. Los bytes de un token pueden ser UTF-8 incompleto.
    Los errores de lectura o de formato se propagan al llamador.
    """
    tokenizador: Any
    decoder: Any
    decodificadores: Any
    agregado: Any
    excluidos: set[int]
    bytes_directos: list[int]
    mapa_bytes: dict[str, int]
    siguiente_codigo: int
    byte: int
    caracter: str
    token: str
    identificador: int
    contenido: bytearray
    tokens: dict[int, bytes]
    gemma: bool

    tokenizador = read_json(modelo.get_path_to_tokenizer_file())
    if not isinstance(tokenizador, dict):
        raise ValueError("El tokenizador debe ser un objeto JSON.")
    decoder = tokenizador.get("decoder")
    if isinstance(decoder, dict) and decoder.get("type") == "Sequence":
        decodificadores = decoder.get("decoders")
        if isinstance(decodificadores, list) and len(decodificadores) == 1:
            decoder = decodificadores[0]
    gemma = decoder == {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    if not gemma and (
        not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel"
    ):
        raise ValueError("Decoder no admitido: se necesita ByteLevel o Gemma.")
    if not vocabulario:
        raise ValueError("El vocabulario no puede estar vacío.")
    if tokenizador.get("model", {}).get("vocab") != vocabulario:
        raise ValueError("El vocabulario no coincide con el del tokenizador.")

    excluidos = set()
    for agregado in tokenizador.get("added_tokens", []):
        excluidos.add(agregado["id"])

    # ByteLevel conserva estos caracteres y representa el resto desde U+0100.
    bytes_directos = list(range(33, 127))
    bytes_directos.extend(range(161, 173))
    bytes_directos.extend(range(174, 256))
    mapa_bytes = {}
    siguiente_codigo = 256
    for byte in range(256):
        if byte in bytes_directos:
            caracter = chr(byte)
        else:
            caracter = chr(siguiente_codigo)
            siguiente_codigo += 1
        mapa_bytes[caracter] = byte

    tokens = {}
    for token, identificador in vocabulario.items():
        if type(identificador) is not int or identificador < 0:
            raise ValueError(f"ID no válido para el token '{token}'.")
        if identificador in excluidos or not token:
            continue
        if identificador in tokens:
            raise ValueError(f"ID de token duplicado: {identificador}.")

        if gemma:
            if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token):
                tokens[identificador] = bytes([int(token[3:5], 16)])
            else:
                tokens[identificador] = token.replace("▁", " ").encode("utf-8")
            continue

        contenido = bytearray()
        for caracter in token:
            if caracter not in mapa_bytes:
                raise ValueError(
                    f"Token incompatible con ByteLevel: {token!r}."
                )
            contenido.append(mapa_bytes[caracter])
        tokens[identificador] = bytes(contenido)

    return tokens


def verificar_tokens(modelo: Any, tokens: dict[int, bytes]) -> None:
    """Contrasta el mapa de bytes con encode() y decode() del SDK."""
    muestra: str
    ids: list[int]
    contenido: bytes

    muestra = '{"texto":"Hola, ñ 日本 😀", "escape":"\\n\\\\\\\""}'
    ids = modelo.encode(muestra).tolist()[0]
    contenido = b"".join(tokens[identificador] for identificador in ids)
    if contenido.decode("utf-8") != muestra or modelo.decode(ids) != muestra:
        raise ValueError("Los bytes de los tokens no coinciden con el SDK.")


@lru_cache(maxsize=262144)
def decodificar_bytes(contenido: bytes) -> tuple[str, bytes]:
    """Decodifica UTF-8 estricto y conserva los bytes aún incompletos."""
    decoder: Any
    texto: str
    pendientes: bytes

    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    texto = decoder.decode(contenido, final=False)
    pendientes = decoder.getstate()[0]
    return texto, pendientes


def cadena_abierta(respuesta: str, restricciones: Restricciones) -> bool:
    """Indica si podemos añadir contenido normal a un valor de cadena."""
    estado: str
    tipo: str
    fragmento: str

    for etapas in restricciones:
        estado, tipo, fragmento = revisar_etapas(respuesta, etapas)
        if estado == "incompleta" and tipo == "string":
            if fragmento.startswith('"') and valor_completo(
                    fragmento + '"', "string"):
                return True
    return False


def enmascarar_logits(
    logits: list[float],
    tokens: dict[int, bytes],
    respuesta: str,
    pendientes: bytes,
    restricciones: Restricciones,
) -> int:
    """Pone a -inf todos los candidatos inválidos antes de seleccionarlos.

    Comprueba el contenido entero del token, incluso si cruza campos.
    Reutiliza prefijos iniciales y permite texto simple dentro de cadenas.
    """
    identificador: int
    logit: float
    contenido: bytes | None
    fragmento: str
    resto: bytes
    candidato: str
    valido: bool
    dentro_cadena: bool
    primeros: dict[str, bool]
    permitidos: int

    dentro_cadena = cadena_abierta(respuesta, restricciones)
    primeros = {}
    permitidos = 0
    for identificador, logit in enumerate(logits):
        contenido = tokens.get(identificador)
        if not isfinite(logit) or not contenido:
            logits[identificador] = float("-inf")
            continue
        try:
            fragmento, resto = decodificar_bytes(pendientes + contenido)
        except UnicodeDecodeError:
            logits[identificador] = float("-inf")
            continue

        if dentro_cadena and not ESPECIALES.search(fragmento):
            valido = True
        else:
            valido = True
            if fragmento:
                if fragmento[0] not in primeros:
                    primeros[fragmento[0]] = estado_restricciones(
                        respuesta + fragmento[0], restricciones
                    ) != "invalida"
                valido = primeros[fragmento[0]]
            if valido:
                candidato = respuesta + fragmento
                # Un carácter pendiente solo cabe como contenido de cadena.
                if resto:
                    candidato += "á"
                valido = estado_restricciones(
                    candidato, restricciones
                ) != "invalida"

        if valido:
            permitidos += 1
        else:
            logits[identificador] = float("-inf")
    return permitidos


def seleccionar_token(logits: list[float]) -> int:
    """Elige el mayor logit finito o falla si no quedan candidatos."""
    siguiente_id: int
    mejor_logit: float
    identificador: int
    logit: float

    siguiente_id = -1
    mejor_logit = float("-inf")
    for identificador, logit in enumerate(logits):
        if isfinite(logit) and logit > mejor_logit:
            siguiente_id = identificador
            mejor_logit = logit
    if siguiente_id == -1:
        raise ValueError("No hay tokens permitidos con logits finitos.")
    return siguiente_id
