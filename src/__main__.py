import json
import sys
from argparse import Namespace
from time import perf_counter
from typing import Any

from .argumentos import leer_argumentos
from .decodificacion import preparar_tokens, seleccionar_token
from .error import (
    ErrorCargaModelo,
    ErrorCargaVocabulario,
    ErrorProcesamientoPromts,
)
from .modelo import cargar_vocabulario, inicializar_modelo
from .parseo import load_functions, load_prompts
from .prompts import construir_prompt
from .salida import guardar_resultados


def generar_respuesta(
    modelo: Any,
    ids: list[int],
    funciones: list[dict[str, Any]],
    tokens_decodificados: dict[int, str],
) -> str:
    """Elige tokens válidos hasta que json.loads() obtiene un objeto."""
    logits: list[float]
    ids_generados: list[int]
    siguiente_id: int
    respuesta: str
    resultado: Any

    ids_generados = []
    respuesta = ""
    resultado = None

    while not isinstance(resultado, dict):
        logits = modelo.get_logits_from_input_ids(ids)

        if not logits:
            raise ValueError("El modelo no ha devuelto logits.")

        siguiente_id = seleccionar_token(
            logits, respuesta, tokens_decodificados, funciones
        )
        ids.append(siguiente_id)
        ids_generados.append(siguiente_id)

        respuesta += tokens_decodificados[siguiente_id]

        try:
            resultado = json.loads(respuesta)
        except json.JSONDecodeError:
            continue

    if modelo.decode(ids_generados) != respuesta:
        raise ValueError("El texto concatenado no coincide con el SDK.")
    return respuesta


def procesar_promts(
    modelo: Any,
    funciones: list[dict[str, Any]],
    promts: list[dict[str, Any]],
    tokens_decodificados: dict[int, str],
) -> list[dict[str, Any]]:
    """Genera y devuelve los resultados reutilizando el mismo modelo."""
    promt: dict[str, Any]
    indice: int
    texto: str
    tokens: Any
    listas_ids: list[list[int]]
    ids: list[int]
    respuesta: str
    llamada: dict[str, Any]
    resultado: dict[str, Any]
    resultados: list[dict[str, Any]]
    inicio: float
    duracion: float

    resultados = []
    for indice, promt in enumerate(promts, start=1):
        try:
            print(
                f"Procesando petición {indice}/{len(promts)}...",
                flush=True,
            )
            inicio = perf_counter()
            texto = construir_prompt(funciones, promt["prompt"])

            tokens = modelo.encode(texto)
            listas_ids = tokens.tolist()
            ids = listas_ids[0]

            respuesta = generar_respuesta(
                modelo, ids, funciones, tokens_decodificados
            )
            llamada = json.loads(respuesta)
            resultado = {
                "prompt": promt["prompt"],
                "name": llamada["name"],
                "parameters": llamada["parameters"],
            }
            resultados.append(resultado)
            duracion = perf_counter() - inicio
            print(
                f"Petición {indice}/{len(promts)} completada "
                f"en {duracion:.2f} s.",
                flush=True,
            )
        except Exception as error:
            raise ErrorProcesamientoPromts(
                f"Petición {indice}: {error}"
            ) from error

    return resultados


def main() -> int:
    """Lee las entradas y guarda las llamadas generadas con restricciones."""
    argumentos: Namespace
    funciones: list[dict[str, Any]]
    promts: list[dict[str, Any]]
    modelo: Any
    vocabulario: dict[str, int]
    tokens_decodificados: dict[int, str]
    resultados: list[dict[str, Any]]
    inicio: float
    duracion: float

    try:
        argumentos = leer_argumentos()
        funciones = load_functions(argumentos.functions_definition)
        promts = load_prompts(argumentos.input)

        if promts and not funciones:
            raise ValueError(
                "Hay peticiones, pero no hay funciones disponibles."
            )

        inicio = perf_counter()
        print("Cargando el modelo...", flush=True)
        modelo = inicializar_modelo()
        vocabulario = cargar_vocabulario(modelo)
        tokens_decodificados = preparar_tokens(modelo, vocabulario)
        print("Modelo preparado.", flush=True)

        resultados = procesar_promts(
            modelo, funciones, promts, tokens_decodificados
        )
        guardar_resultados(argumentos.output, resultados)
        duracion = perf_counter() - inicio
        print(f"Resultados guardados en {argumentos.output}", flush=True)
        print(
            f"Tiempo total: {duracion:.2f} s.",
            flush=True,
        )
    except ErrorCargaModelo as error:
        print(f"No se pudo cargar el modelo: {error}", file=sys.stderr)
        return 1
    except ErrorCargaVocabulario as error:
        print(
            f"No se pudo cargar el vocabulario: {error}", file=sys.stderr
        )
        return 1
    except ErrorProcesamientoPromts as error:
        print(
            f"No se pudieron procesar los prompts: {error}", file=sys.stderr
        )
        return 1
    except OSError as error:
        print(
            f"No se pudo leer o guardar un archivo: {error}", file=sys.stderr
        )
        return 1
    except ValueError as error:
        print(f"Contenido o valor no válido: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Ejecución interrumpida por el usuario.", file=sys.stderr)
        return 130
    except Exception as error:
        print(
            f"No se pudo completar la carga o procesar los prompts: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
