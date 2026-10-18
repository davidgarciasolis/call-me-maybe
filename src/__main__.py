import json
import sys
from time import perf_counter
from typing import Any

from .parseo import load_functions, load_prompts
from .prompts import construir_prompt


def main() -> int:
    """Lee las entradas y muestra una generación de prueba sin restricciones."""
    funciones: list[dict[str, Any]]
    promts: list[dict[str, Any]]
    promt: dict[str, Any]
    indice: int
    texto: str
    tokens: Any
    listas_ids: list[list[int]]
    ids: list[int]
    logits: list[float]
    ids_generados: list[int]
    siguiente_id: int
    identificador: int
    logit: float
    paso: int
    limite_tokens: int
    respuesta: str
    resultado: Any
    inicio: float
    duracion: float
    modelo: Any
    vocabulario: dict[str, int]
    error: Exception

    try:
        funciones = load_functions("data/input/functions_definition.json")
        promts = load_prompts("data/input/function_calling_tests.json")
    except OSError as error:
        print(f"No se pudo leer un archivo: {error}", file=sys.stderr)
        return 1
    except (ValueError, UnicodeDecodeError) as error:
        print(f"Contenido del archivo no válido: {error}", file=sys.stderr)
        return 1

    print("Funciones:")
    print(funciones)

    print("Peticiones:")
    print(promts)

    try:
        from .modelo import cargar_vocabulario, inicializar_modelo

        print("Cargando el modelo...")
        modelo = inicializar_modelo()
        vocabulario = cargar_vocabulario(modelo)
    except ImportError as error:
        print(
            f"No se pudo importar el SDK o sus dependencias: {error}. "
            "Ejecuta 'uv sync' y utiliza 'uv run python -m src'.",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(
            f"No se pudo cargar el modelo o su vocabulario: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"Modelo preparado. Vocabulario: {len(vocabulario)} tokens.")

    try:
        print("Prompts preparados para el modelo:")
        for indice, promt in enumerate(promts, start=1):
            texto = construir_prompt(funciones, promt["prompt"])
            print(f"\n========== Prompt {indice} ==========\n")
            print(texto)

            tokens = modelo.encode(texto)
            listas_ids = tokens.tolist()
            ids = listas_ids[0]

            print("IDs del prompt:")
            print(ids)

            limite_tokens = 50
            ids_generados = []
            print("Generando una vista previa sin restricciones...", flush=True)

            for paso in range(limite_tokens):
                print(
                    f"Calculando token {paso + 1}/{limite_tokens}...",
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

                try:
                    resultado = json.loads(respuesta)
                except json.JSONDecodeError:
                    continue

                if isinstance(resultado, dict):
                    print("JSON completo. Generación terminada.", flush=True)
                    break
            else:
                print(
                    "Límite de tokens alcanzado sin completar un objeto JSON.",
                    flush=True,
                )

            respuesta = modelo.decode(ids_generados)
            print("Respuesta sin restricciones:")
            print(respuesta)
    except Exception as error:
        print(f"No se pudieron procesar los prompts: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
