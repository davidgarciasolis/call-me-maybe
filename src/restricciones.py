"""Etapas y progreso de las restricciones durante la generación."""

import json
from typing import Any

from .validacion import prefijo_valor_valido, valor_completo

# Etapa, posición en texto fijo, valor numérico/booleano y estado de cadena.
Progreso = tuple[int, int, str, str]


def avanzar(
    progreso: Progreso,
    etapas: list[tuple[str, str]],
    fragmento: str,
) -> Progreso | None:
    """Prueba un fragmento sin modificar el progreso recibido.

    Devuelve None si no puede completar la función. Solo revisa caracteres
    nuevos de las cadenas. Para números y booleanos revisa el valor actual.
    No vuelve a recorrer las etapas anteriores.
    """
    etapa: int
    posicion: int
    valor: str
    cadena: str
    indice: int
    caracter: str
    tipo: str
    texto: str

    etapa, posicion, valor, cadena = progreso
    indice = 0
    while indice < len(fragmento):
        if etapa == len(etapas):
            return None
        tipo, texto = etapas[etapa]
        caracter = fragmento[indice]

        if tipo == "fijo":
            if caracter != texto[posicion]:
                return None
            posicion += 1
            if posicion == len(texto):
                etapa += 1
                posicion = 0
        elif tipo == "string":
            if cadena == "cerrada":
                if caracter not in ",}":
                    return None
                etapa += 1
                cadena = "inicio"
                # El separador pertenece a la siguiente etapa.
                continue
            if cadena == "inicio":
                if caracter != '"':
                    return None
                cadena = "abierta"
            elif cadena == "escape":
                if caracter not in '"\\/bfnrt':
                    return None
                cadena = "abierta"
            elif caracter == "\\":
                cadena = "escape"
            elif caracter == '"':
                cadena = "cerrada"
            elif ord(caracter) < 32:
                return None
        else:
            if caracter in ",}" and valor_completo(valor, tipo):
                etapa += 1
                valor = ""
                # El separador pertenece a la siguiente etapa.
                continue
            valor += caracter
            if not prefijo_valor_valido(valor, tipo):
                return None
        indice += 1

    return etapa, posicion, valor, cadena


def avanzar_funciones(
    progresos: list[Progreso | None],
    etapas_funciones: list[list[tuple[str, str]]],
    fragmento: str,
) -> list[Progreso | None]:
    """Devuelve los progresos nuevos y mantiene las funciones descartadas."""
    nuevos: list[Progreso | None]
    progreso: Progreso | None
    etapas: list[tuple[str, str]]

    nuevos = []
    for progreso, etapas in zip(progresos, etapas_funciones):
        if progreso is None:
            nuevos.append(None)
        else:
            nuevos.append(avanzar(progreso, etapas, fragmento))
    return nuevos


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
        if definicion["type"] not in ("string", "number", "boolean"):
            raise ValueError(
                f"Tipo no admitido: '{definicion['type']}'."
            )
        etapas.append(("fijo", separador + json.dumps(parametro) + ":"))
        etapas.append((definicion["type"], ""))
        separador = ","
    etapas.append(("fijo", "}}"))

    return etapas


def estado_llamada(
    respuesta: str,
    funciones: list[dict[str, Any]],
) -> str:
    """Devuelve 'invalida', 'incompleta' o 'completa' para una llamada.

    Revisa desde el principio JSON compacto, con name antes de parameters
    y los parámetros en el orden de las definiciones. Recibe funciones
    validadas por load_functions() y admite string, number y boolean.
    """
    etapas_funciones: list[list[tuple[str, str]]]

    etapas_funciones = [preparar_etapas(funcion) for funcion in funciones]
    return estado_etapas(respuesta, etapas_funciones)


def estado_etapas(
    respuesta: str,
    etapas_funciones: list[list[tuple[str, str]]],
) -> str:
    """Valida desde el inicio con el mismo avance que utiliza la máscara."""
    etapas: list[tuple[str, str]]
    progreso: Progreso | None
    hay_incompleta: bool

    hay_incompleta = False
    for etapas in etapas_funciones:
        progreso = avanzar((0, 0, "", "inicio"), etapas, respuesta)
        if progreso is None:
            continue
        if progreso[0] == len(etapas):
            return "completa"
        hay_incompleta = True

    if hay_incompleta:
        return "incompleta"
    return "invalida"
