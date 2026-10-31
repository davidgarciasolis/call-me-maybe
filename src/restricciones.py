"""Validación de llamadas frente a las definiciones de funciones."""

from math import isfinite
from typing import Any


def prefijo_cadena_valido(fragmento: str) -> bool:
    """Indica si el fragmento puede formar una cadena JSON válida.

    Recibe solo la cadena, incluida su comilla inicial, sin separadores
    ni espacios exteriores. Acepta el fragmento vacío, cadenas completas
    y cadenas incompletas, incluso con escapes pendientes de terminar.
    Admite escapes básicos de JSON, pero rechaza los escapes Unicode.
    """
    indice: int
    caracter: str
    escape: bool

    if not fragmento:
        return True
    if fragmento[0] != '"':
        return False

    escape = False

    for indice, caracter in enumerate(fragmento[1:], start=1):
        if escape:
            if caracter not in '"\\/bfnrt':
                return False
            escape = False
        elif caracter == "\\":
            escape = True
        elif caracter == '"':
            return indice == len(fragmento) - 1
        elif ord(caracter) < 32:
            return False

    return True


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
    if "true".startswith(fragmento):
        return True

    if "false".startswith(fragmento):
        return True

    return False


def prefijo_valor_valido(fragmento: str, tipo: str) -> bool:
    """Comprueba un prefijo según el tipo declarado del parámetro."""
    if tipo == "string":
        return prefijo_cadena_valido(fragmento)

    if tipo == "number":
        return prefijo_numero_valido(fragmento)

    if tipo == "boolean":
        return prefijo_booleano_valido(fragmento)

    return False


def validar_llamada(
    llamada: Any,
    funciones: list[dict[str, Any]],
) -> dict[str, Any]:
    """Devuelve una llamada válida o lanza ValueError con el motivo.

    Recibe el objeto obtenido con json.loads() y las definiciones ya
    validadas por load_functions(). Admite parámetros string, number,
    y boolean, sin convertir sus valores.
    """
    nombre: str
    funcion: dict[str, Any]
    parametros: dict[str, Any]
    definiciones: dict[str, Any]
    parametro: str
    valor: Any
    tipo: str
    valido: bool

    if not isinstance(llamada, dict):
        raise ValueError("La llamada debe ser un objeto JSON.")
    if set(llamada) != {"name", "parameters"}:
        raise ValueError(
            "La llamada debe contener exactamente 'name' y 'parameters'."
        )
    if not isinstance(llamada["name"], str):
        raise ValueError("El nombre de la función debe ser texto.")
    if not isinstance(llamada["parameters"], dict):
        raise ValueError("'parameters' debe ser un objeto JSON.")

    nombre = llamada["name"]
    parametros = llamada["parameters"]

    for funcion in funciones:
        if funcion["name"] != nombre:
            continue

        definiciones = funcion["parameters"]
        for parametro in definiciones:
            if parametro not in parametros:
                raise ValueError(
                    f"{nombre}: falta el parámetro '{parametro}'."
                )
        for parametro in parametros:
            if parametro not in definiciones:
                raise ValueError(
                    f"{nombre}: parámetro no definido '{parametro}'."
                )

        for parametro in definiciones:
            tipo = definiciones[parametro]["type"]
            valor = parametros[parametro]

            if tipo == "string":
                valido = isinstance(valor, str)
            elif tipo == "number":
                valido = type(valor) in (int, float)
                if type(valor) is float and not isfinite(valor):
                    valido = False
            elif tipo == "boolean":
                valido = type(valor) is bool
            else:
                raise ValueError(
                    f"{nombre}: tipo no admitido '{tipo}' "
                    f"para el parámetro '{parametro}'."
                )

            if not valido:
                raise ValueError(
                    f"{nombre}: el parámetro '{parametro}' "
                    f"debe ser de tipo '{tipo}'."
                )

        return llamada

    raise ValueError(f"Función no disponible: '{nombre}'.")
