"""Lectura y validación de los archivos de entrada."""

import json
from typing import Any, TextIO


def read_json(ruta: str) -> Any:
    """Lee un archivo JSON y devuelve su contenido como datos de Python."""
    archivo: TextIO

    with open(ruta) as archivo:
        return json.load(archivo)


def validar_campos(funcion: Any, contexto: str) -> str:
    """Valida el nombre y la descripción y devuelve el nombre."""
    campo: str
    nombre: str

    if not isinstance(funcion, dict):
        raise ValueError(f"{contexto}: debe ser un objeto.")

    for campo in ("name", "description"):
        if not isinstance(funcion.get(campo), str):
            raise ValueError(f"{contexto}: '{campo}' debe ser texto.")

    nombre = funcion["name"]
    if not nombre.strip():
        raise ValueError(f"{contexto}: el nombre no puede estar vacío.")

    return nombre


def validar_parametros(funcion: dict[str, Any], contexto: str) -> None:
    """Valida la estructura y el tipo declarado de cada parámetro."""
    parametros: Any
    parametro: str
    definicion: Any
    tipo: Any

    parametros = funcion.get("parameters")
    if not isinstance(parametros, dict):
        raise ValueError(f"{contexto}: 'parameters' debe ser un objeto.")

    for parametro, definicion in parametros.items():
        if not isinstance(definicion, dict):
            raise ValueError(
                f"{contexto}: el parámetro '{parametro}' debe ser un objeto."
            )
        tipo = definicion.get("type")
        if not isinstance(tipo, str) or not tipo.strip():
            raise ValueError(
                f"{contexto}: el parámetro '{parametro}' necesita un 'type' "
                "de texto no vacío."
            )


def validar_retorno(funcion: dict[str, Any], contexto: str) -> None:
    """Valida la estructura y el tipo declarado del retorno."""
    retorno: Any
    tipo_retorno: Any

    retorno = funcion.get("returns")
    if not isinstance(retorno, dict):
        raise ValueError(f"{contexto}: 'returns' debe ser un objeto.")
    tipo_retorno = retorno.get("type")
    if not isinstance(tipo_retorno, str) or not tipo_retorno.strip():
        raise ValueError(
            f"{contexto}: 'returns.type' debe ser texto no vacío."
        )


def load_functions(ruta: str) -> list[dict[str, Any]]:
    """Lee y valida las definiciones de funciones del archivo indicado."""
    funciones: Any
    nombres: set[str]
    indice: int
    funcion: Any
    contexto: str
    nombre: str

    funciones = read_json(ruta)

    if not isinstance(funciones, list):
        raise ValueError(f"{ruta}: se esperaba una lista de funciones.")

    nombres = set()
    for indice, funcion in enumerate(funciones, start=1):
        contexto = f"{ruta}: función {indice}"
        nombre = validar_campos(funcion, contexto)
        if nombre in nombres:
            raise ValueError(f"{contexto}: nombre duplicado '{nombre}'.")
        nombres.add(nombre)

        validar_parametros(funcion, contexto)
        validar_retorno(funcion, contexto)

    return funciones


def load_prompts(ruta: str) -> list[dict[str, Any]]:
    """Lee y valida una lista de objetos con un campo 'prompt' de texto."""
    promts: Any
    indice: int
    promt: Any
    contexto: str

    promts = read_json(ruta)

    if not isinstance(promts, list):
        raise ValueError(f"{ruta}: se esperaba una lista de promts.")

    for indice, promt in enumerate(promts, start=1):
        contexto = f"{ruta}: promt {indice}"
        if not isinstance(promt, dict):
            raise ValueError(f"{contexto}: debe ser un objeto.")
        if not isinstance(promt.get("prompt"), str):
            raise ValueError(f"{contexto}: 'prompt' debe ser texto.")

    return promts
