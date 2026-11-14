"""Validación de la respuesta completa desde el principio, sin etapas."""

import json
from typing import Any

from .validacion import prefijo_valor_valido, valor_completo


def consumir_fijo(
    respuesta: str, posicion: int, esperado: str,
) -> int | None:
    """Devuelve la posición prevista, o None si el texto no coincide.

    Si supera la longitud de respuesta, el texto fijo está incompleto.
    """
    fragmento: str

    fragmento = respuesta[posicion:posicion + len(esperado)]
    if not esperado.startswith(fragmento):
        return None
    return posicion + len(esperado)


def consumir_cadena(respuesta: str, posicion: int) -> int | None:
    """Recorre una cadena; devuelve el final disponible o None si es inválida.

    Admite cierre o escape pendientes al final del texto disponible.
    Rechaza las cadenas vacías, pero permite espacios como contenido.
    Conserva los escapes simples admitidos por la versión anterior.
    """
    caracter: str

    if respuesta[posicion] != '"':
        return None
    if respuesta[posicion:posicion + 2] == '""':
        return None
    posicion += 1
    while posicion < len(respuesta):
        caracter = respuesta[posicion]
        posicion += 1
        if caracter == '"':
            return posicion
        elif ord(caracter) < 32:
            return None
    return posicion


def prefijo_funcion_valido(
    respuesta: str, funcion: dict[str, Any],
) -> bool:
    """Revisa toda la respuesta según una función, sin guardar progreso.

    Exige JSON compacto, parámetros en orden y tipos string/number/boolean.
    Acepta prefijos incompletos que todavía pueden formar la llamada.
    """
    posicion: int | None
    inicio: int
    nombre: str
    definicion: dict[str, Any]
    tipo: str
    valor: str
    separador: str

    posicion = consumir_fijo(
        respuesta, 0,
        '{"name":' + json.dumps(funcion["name"]) + ',"parameters":{',
    )
    if posicion is None:
        return False
    if posicion >= len(respuesta):
        return True

    separador = ""
    for nombre, definicion in funcion["parameters"].items():
        posicion = consumir_fijo(
            respuesta, posicion, separador + json.dumps(nombre) + ":",
        )
        if posicion is None:
            return False
        if posicion >= len(respuesta):
            return True

        inicio = posicion
        tipo = definicion["type"]
        if tipo == "string":
            posicion = consumir_cadena(respuesta, posicion)
            if posicion is None:
                return False
            if posicion == len(respuesta):
                return True
        else:
            while (posicion < len(respuesta)
                   and respuesta[posicion] not in ",}"):
                posicion += 1
            valor = respuesta[inicio:posicion]
            if posicion == len(respuesta):
                return prefijo_valor_valido(valor, tipo)
            if not valor_completo(valor, tipo):
                return False
        separador = ","

    posicion = consumir_fijo(respuesta, posicion, "}}")
    return posicion is not None and posicion >= len(respuesta)


def prefijo_llamada_valido(
    respuesta: str, funciones: list[dict[str, Any]],
) -> bool:
    """Acepta la respuesta si puede completar alguna función disponible."""
    funcion: dict[str, Any]

    for funcion in funciones:
        if prefijo_funcion_valido(respuesta, funcion):
            return True
    return False
