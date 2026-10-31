"""Validación de llamadas frente a las definiciones de funciones."""

import json
from math import isfinite
from typing import Any

Etapas = tuple[tuple[str, str], ...]
Restricciones = tuple[Etapas, ...]


def prefijo_cadena_valido(fragmento: str) -> bool:
    """Indica si el fragmento puede formar una cadena JSON válida.

    Recibe solo la cadena, incluida su comilla inicial, sin separadores
    ni espacios exteriores. Acepta el fragmento vacío, cadenas completas
    y cadenas incompletas, incluso con escapes pendientes de terminar.
    Admite escapes básicos de JSON y escapes Unicode de cuatro dígitos.
    """
    indice: int
    caracter: str
    escape: bool
    unicode_pendiente: int

    if not fragmento:
        return True
    if fragmento[0] != '"':
        return False

    escape = False
    unicode_pendiente = 0

    for indice, caracter in enumerate(fragmento[1:], start=1):
        if unicode_pendiente:
            if caracter not in "0123456789abcdefABCDEF":
                return False
            unicode_pendiente -= 1
        elif escape:
            if caracter == "u":
                unicode_pendiente = 4
            elif caracter not in '"\\/bfnrt':
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

    if tipo == "integer":
        return "." not in fragmento and prefijo_numero_valido(fragmento)

    if tipo == "boolean":
        return prefijo_booleano_valido(fragmento)

    if tipo == "null":
        return "null".startswith(fragmento)

    return False


def valor_completo(fragmento: str, tipo: str) -> bool:
    """Comprueba si un valor del tipo declarado está terminado."""
    if not prefijo_valor_valido(fragmento, tipo):
        return False

    try:
        json.loads(fragmento)
    except ValueError:
        return False

    return True


def preparar_etapas(funcion: dict[str, Any]) -> list[tuple[str, str]]:
    """Construye las etapas de texto fijo y valores de una función.

    Cada tupla contiene 'fijo' y el texto esperado, o el tipo de un valor
    y un texto vacío. Conserva el orden de los parámetros de la definición.
    """
    etapas: list[tuple[str, str]]
    parametro: str
    definicion: dict[str, Any]
    separador: str

    etapas = [
        ("fijo", '{"name":' + json.dumps(funcion["name"])
         + ',"parameters":{'),
    ]
    separador = ""
    for parametro, definicion in funcion["parameters"].items():
        if definicion["type"] not in (
            "string", "number", "integer", "boolean", "null",
        ):
            raise ValueError(
                f"Tipo no admitido: '{definicion['type']}'."
            )
        etapas.append(("fijo", separador + json.dumps(parametro) + ":"))
        etapas.append((definicion["type"], ""))
        separador = ","
    etapas.append(("fijo", "}}"))

    return etapas


def consumir_valor(respuesta: str, posicion: int) -> int:
    """Localiza el final de un valor desde su posición inicial.

    Se detiene antes de una coma o llave fuera de una cadena, o al final
    del texto. Respeta comillas y escapes, pero no valida el valor.
    """
    caracter: str
    en_cadena: bool
    escape: bool

    en_cadena = False
    escape = False
    while posicion < len(respuesta):
        caracter = respuesta[posicion]
        if en_cadena:
            if escape:
                escape = False
            elif caracter == "\\":
                escape = True
            elif caracter == '"':
                en_cadena = False
        elif caracter in ",}":
            break
        elif caracter == '"':
            en_cadena = True
        posicion += 1

    return posicion


def preparar_restricciones(
    funciones: list[dict[str, Any]],
) -> Restricciones:
    """Prepara una vez las etapas de todas las funciones disponibles."""
    funcion: dict[str, Any]
    restricciones: list[Etapas]

    restricciones = []
    for funcion in funciones:
        restricciones.append(tuple(preparar_etapas(funcion)))
    return tuple(restricciones)


def revisar_etapas(respuesta: str, etapas: Etapas) -> tuple[str, str, str]:
    """Devuelve el estado y el tipo y fragmento del valor pendiente."""
    tipo: str
    texto: str
    fragmento: str
    posicion: int
    inicio: int

    posicion = 0
    for tipo, texto in etapas:
        if tipo == "fijo":
            fragmento = respuesta[posicion:posicion + len(texto)]
            if not texto.startswith(fragmento):
                return "invalida", "", ""
            if len(fragmento) < len(texto):
                return "incompleta", "fijo", fragmento
            posicion += len(texto)
            continue

        inicio = posicion
        posicion = consumir_valor(respuesta, posicion)
        fragmento = respuesta[inicio:posicion]
        if not prefijo_valor_valido(fragmento, tipo):
            return "invalida", "", ""
        if not valor_completo(fragmento, tipo):
            if posicion == len(respuesta):
                return "incompleta", tipo, fragmento
            return "invalida", "", ""

    if posicion == len(respuesta):
        return "completa", "", ""
    return "invalida", "", ""


def estado_restricciones(respuesta: str, restricciones: Restricciones) -> str:
    """Comprueba una llamada con las etapas ya preparadas."""
    etapas: Etapas
    estado: str
    incompleta: bool

    incompleta = False
    for etapas in restricciones:
        estado = revisar_etapas(respuesta, etapas)[0]
        if estado == "completa":
            return estado
        if estado == "incompleta":
            incompleta = True
    return "incompleta" if incompleta else "invalida"


def estado_llamada(respuesta: str, funciones: list[dict[str, Any]]) -> str:
    """Comprueba una llamada sin reutilizar las etapas entre consultas."""
    return estado_restricciones(respuesta, preparar_restricciones(funciones))


def validar_llamada(
    llamada: Any,
    funciones: list[dict[str, Any]],
) -> dict[str, Any]:
    """Devuelve una llamada válida o lanza ValueError con el motivo.

    Recibe el objeto obtenido con json.loads() y las definiciones ya
    validadas por load_functions(). No convierte los valores recibidos.
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
            elif tipo == "integer":
                valido = type(valor) is int
            elif tipo == "null":
                valido = valor is None
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
