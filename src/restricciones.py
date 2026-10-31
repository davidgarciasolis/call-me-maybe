"""Validación de llamadas frente a las definiciones de funciones."""

from math import isfinite
from typing import Any


def validar_llamada(
    llamada: Any,
    funciones: list[dict[str, Any]],
) -> dict[str, Any]:
    """Devuelve una llamada válida o lanza ValueError con el motivo.

    Recibe el objeto obtenido con json.loads() y las definiciones ya
    validadas por load_functions(). Admite parámetros string, number,
    integer, boolean y null, sin convertir sus valores.
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
            elif tipo == "integer":
                valido = type(valor) is int
            elif tipo == "boolean":
                valido = type(valor) is bool
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
