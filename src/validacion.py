"""Validación de prefijos y valores JSON durante la generación."""

import json


def prefijo_numero_valido(fragmento: str) -> bool:
    """Comprueba un prefijo de número JSON, sin exponentes ni espacios.

    Admite enteros y decimales. El texto vacío, el signo menos aislado
    y un número terminado en punto son prefijos que pueden completarse.
    """
    partes: list[str]
    entero: str
    caracter: str

    if fragmento.startswith("-"):
        fragmento = fragmento[1:]
    if not fragmento:
        return True

    for caracter in fragmento:
        if caracter not in "0123456789.":
            return False

    partes = fragmento.split(".")
    if len(partes) > 2:
        return False

    entero = partes[0]
    if not entero:
        return False
    if len(entero) > 1 and entero[0] == "0":
        return False

    return True


def prefijo_booleano_valido(fragmento: str) -> bool:
    """Indica si el fragmento puede formar true o false, incluido el vacío."""
    return "true".startswith(fragmento) or "false".startswith(fragmento)


def prefijo_valor_valido(fragmento: str, tipo: str) -> bool:
    """Comprueba un prefijo según el tipo declarado del parámetro."""
    if tipo == "string":
        if not fragmento:
            return True
        try:
            return isinstance(json.loads(fragmento), str)
        except json.JSONDecodeError as error:
            return error.msg == "Unterminated string starting at"

    if tipo == "number":
        return prefijo_numero_valido(fragmento)

    if tipo == "boolean":
        return prefijo_booleano_valido(fragmento)

    return False


def valor_completo(fragmento: str, tipo: str) -> bool:
    """Comprueba si un prefijo válido ya es un valor JSON completo."""
    if not prefijo_valor_valido(fragmento, tipo):
        return False

    try:
        json.loads(fragmento)
    except ValueError:
        return False

    return True
