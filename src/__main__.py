import json
import sys
from time import perf_counter
from typing import Any

from .error import (
    ErrorCargaModelo,
    ErrorCargaVocabulario,
    ErrorProcesamientoPromts,
)
from .modelo import cargar_vocabulario, inicializar_modelo
from .parseo import load_functions, load_prompts
from .prompts import construir_prompt


def generar_respuesta(modelo: Any, ids: list[int]) -> str:
    """Genera texto sin restricciones y añade sus tokens a los IDs recibidos."""
    logits: list[float]
    ids_generados: list[int]
    siguiente_id: int
    identificador: int
    logit: float
    paso: int
    respuesta: str
    resultado: Any
    inicio: float
    duracion: float

    paso = 0
    ids_generados = []
    resultado = None
    print("Generando una vista previa sin restricciones...", flush=True)

    while not isinstance(resultado, dict):
        print(
            f"Calculando token {paso + 1}...",
            flush=True,
        )
        inicio = perf_counter()
        logits = modelo.get_logits_from_input_ids(ids)
        duracion = perf_counter() - inicio
        print(f"Consulta completada en {duracion:.2f} s.", flush=True)

        if paso == 0:
            print("Cantidad de logits:")
            print(len(logits))
            print("Primeros 10 logits (por ID):")
            print(logits[:10])

        if not logits:
            raise ValueError("El modelo no ha devuelto logits.")

        siguiente_id = 0
        for identificador, logit in enumerate(logits):
            if logit > logits[siguiente_id]:
                siguiente_id = identificador

        ids.append(siguiente_id)
        ids_generados.append(siguiente_id)

        respuesta = modelo.decode(ids_generados)
        print("Respuesta parcial:", repr(respuesta), flush=True)
        paso += 1

        try:
            resultado = json.loads(respuesta)
        except json.JSONDecodeError:
            continue

    print("JSON completo. Generación terminada.", flush=True)
    respuesta = modelo.decode(ids_generados)
    return respuesta


def procesar_promts(
    modelo: Any,
    funciones: list[dict[str, Any]],
    promts: list[dict[str, Any]],
) -> None:
    """Prepara y genera una respuesta para cada petición con el mismo modelo."""
    promt: dict[str, Any]
    indice: int
    texto: str
    tokens: Any
    listas_ids: list[list[int]]
    ids: list[int]
    respuesta: str
    llamada: dict[str, Any]
    resultado: dict[str, Any]
    error: Exception

    print("Prompts preparados para el modelo:")
    for indice, promt in enumerate(promts, start=1):
        try:
            texto = construir_prompt(funciones, promt["prompt"])
            print(f"\n========== Prompt {indice} ==========\n")
            print(texto)

            tokens = modelo.encode(texto)
            listas_ids = tokens.tolist()
            ids = listas_ids[0]

            print("IDs del prompt:")
            print(ids)

            respuesta = generar_respuesta(modelo, ids)
            llamada = json.loads(respuesta)
            resultado = {
                "prompt": promt["prompt"],
                "name": llamada["name"],
                "parameters": llamada["parameters"],
            }
            print("Resultado:")
            print(json.dumps(resultado, ensure_ascii=False, indent=4))
        except Exception as error:
            raise ErrorProcesamientoPromts(
                f"Petición {indice}: {error}"
            ) from error


def main() -> int:
    """Lee las entradas y muestra una generación de prueba sin restricciones."""
    funciones: list[dict[str, Any]]
    promts: list[dict[str, Any]]
    modelo: Any
    vocabulario: dict[str, int]
    error: Exception

    try:
        funciones = load_functions("data/input/functions_definition.json")
        promts = load_prompts("data/input/function_calling_tests.json")

        print("Funciones:")
        print(funciones)

        print("Peticiones:")
        print(promts)

        print("Cargando el modelo...")
        modelo = inicializar_modelo()
        vocabulario = cargar_vocabulario(modelo)
        print(f"Modelo preparado. Vocabulario: {len(vocabulario)} tokens.")

        procesar_promts(modelo, funciones, promts)
    except ErrorCargaModelo as error:
        print(f"No se pudo cargar el modelo: {error}", file=sys.stderr)
        return 1
    except ErrorCargaVocabulario as error:
        print(f"No se pudo cargar el vocabulario: {error}", file=sys.stderr)
        return 1
    except ErrorProcesamientoPromts as error:
        print(f"No se pudieron procesar los prompts: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"No se pudo acceder a un archivo: {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Contenido o valor no válido: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"No se pudo completar la carga o procesar los prompts: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
