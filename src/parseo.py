"""Lectura y validación de los archivos de entrada."""

import json
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ValidationError, field_validator


class Parametro(BaseModel):
    """Definición de un parámetro con uno de los tipos admitidos."""

    type: Literal["string", "number", "boolean"]


class Retorno(BaseModel):
    """Definición del tipo de retorno de una función."""

    type: str

    @field_validator("type")
    @classmethod
    def validar_tipo(cls, valor: str) -> str:
        """Rechaza un tipo vacío o compuesto solo por espacios."""
        if not valor.strip():
            raise ValueError("El tipo de retorno no puede estar vacío.")
        return valor


class Funcion(BaseModel):
    """Estructura de una función disponible para el modelo."""

    name: str
    description: str
    parameters: dict[str, Parametro]
    returns: Retorno

    @field_validator("name")
    @classmethod
    def validar_nombre(cls, valor: str) -> str:
        """Rechaza un nombre vacío o compuesto solo por espacios."""
        if not valor.strip():
            raise ValueError("El nombre no puede estar vacío.")
        return valor


class Peticion(BaseModel):
    """Petición en lenguaje natural que contiene un campo prompt."""

    prompt: str


def read_json(ruta: str) -> Any:
    """Lee un archivo JSON y devuelve su contenido como datos de Python."""
    archivo: TextIO

    with open(ruta) as archivo:
        return json.load(archivo)


def load_functions(ruta: str) -> list[dict[str, Any]]:
    """Lee y valida las funciones conservando los diccionarios originales."""
    funciones: Any
    nombres: set[str]
    indice: int
    funcion: Any
    contexto: str
    validada: Funcion

    funciones = read_json(ruta)
    if not isinstance(funciones, list):
        raise ValueError(f"{ruta}: se esperaba una lista de funciones.")

    nombres = set()
    for indice, funcion in enumerate(funciones, start=1):
        contexto = f"{ruta}: función {indice}"
        try:
            validada = Funcion.model_validate(funcion, strict=True)
        except ValidationError as error:
            raise ValueError(f"{contexto}: {error}") from error

        if validada.name in nombres:
            raise ValueError(
                f"{contexto}: nombre duplicado '{validada.name}'."
            )
        nombres.add(validada.name)

    return funciones


def load_prompts(ruta: str) -> list[dict[str, Any]]:
    """Lee y valida las peticiones conservando los diccionarios originales."""
    promts: Any
    indice: int
    promt: Any
    contexto: str

    promts = read_json(ruta)
    if not isinstance(promts, list):
        raise ValueError(f"{ruta}: se esperaba una lista de promts.")

    for indice, promt in enumerate(promts, start=1):
        contexto = f"{ruta}: promt {indice}"
        try:
            Peticion.model_validate(promt, strict=True)
        except ValidationError as error:
            raise ValueError(f"{contexto}: {error}") from error

    return promts
