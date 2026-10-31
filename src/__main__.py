"""Punto de entrada para generar llamadas con decodificación restringida."""

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

# También se aplica al ejecutar directamente uv run python -m src.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from .decodificacion import (  # noqa: E402
    decodificar_bytes,
    enmascarar_logits,
    preparar_tokens,
    seleccionar_token,
    verificar_tokens,
)
from .error import (  # noqa: E402
    ErrorCargaModelo,
    ErrorCargaVocabulario,
    ErrorProcesamientoPromts,
)
from .modelo import cargar_vocabulario, inicializar_modelo  # noqa: E402
from .parseo import load_functions, load_prompts  # noqa: E402
from .prompts import construir_prompt  # noqa: E402
from .restricciones import (  # noqa: E402
    Restricciones,
    estado_restricciones,
    preparar_restricciones,
    validar_llamada,
)


def leer_argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    """Lee las rutas de entrada y salida y la opción de progreso detallado."""
    parser: argparse.ArgumentParser

    parser = argparse.ArgumentParser(
        description="Traduce peticiones a llamadas de funciones con un LLM."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output", default="data/output/function_calling_results.json",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--model", choices=[
            "Qwen/Qwen3-0.6B", "google/gemma-3-270m-it",
            "openai-community/gpt2",
            "LiquidAI/LFM2.5-230M",
        ],
        default="Qwen/Qwen3-0.6B",
    )
    return parser.parse_args(argv)


def generar_respuesta(
    modelo: Any,
    ids: list[int],
    tokens: dict[int, bytes],
    restricciones: Restricciones,
    verbose: bool = False,
) -> str:
    """Enmascara antes de elegir cada token y para al completar la llamada.

    Reutiliza el contexto recibido. No impone un límite de tokens: una
    generación que nunca cierre un valor puede interrumpirse con Ctrl+C.
    """
    logits: list[float]
    ids_generados: list[int]
    siguiente_id: int
    respuesta: str
    fragmento: str
    pendientes: bytes
    permitidos: int
    inicio: float
    inferencia: float
    mascara: float

    respuesta = ""
    pendientes = b""
    ids_generados = []
    if not restricciones:
        raise ValueError(
            "No hay funciones disponibles para generar una llamada."
        )

    while pendientes or estado_restricciones(
            respuesta, restricciones) != "completa":
        inicio = perf_counter()
        logits = modelo.get_logits_from_input_ids(ids)
        inferencia = perf_counter() - inicio
        if not logits:
            raise ValueError("El modelo no ha devuelto logits.")
        inicio = perf_counter()
        permitidos = enmascarar_logits(
            logits, tokens, respuesta, pendientes, restricciones,
        )
        mascara = perf_counter() - inicio
        siguiente_id = seleccionar_token(logits)
        ids.append(siguiente_id)
        ids_generados.append(siguiente_id)
        fragmento, pendientes = decodificar_bytes(
            pendientes + tokens[siguiente_id]
        )
        respuesta += fragmento
        if verbose:
            print(
                f"Token {len(ids_generados)}; permitidos: {permitidos}; "
                f"inferencia: {inferencia:.2f} s; máscara: {mascara:.2f} s; "
                f"respuesta: {respuesta!r}", flush=True,
            )

    if modelo.decode(ids_generados) != respuesta:
        raise ValueError("La respuesta no coincide con decode() del SDK.")
    return respuesta


def procesar_promts(
    modelo: Any,
    funciones: list[dict[str, Any]],
    promts: list[dict[str, Any]],
    tokens: dict[int, bytes],
    restricciones: Restricciones,
    verbose: bool = False,
    nombre_modelo: str = "Qwen/Qwen3-0.6B",
) -> list[dict[str, Any]]:
    """Genera y valida las llamadas reutilizando el modelo, sin ejecutarlas."""
    promt: dict[str, Any]
    indice: int
    texto: str
    ids: list[int]
    respuesta: str
    llamada: dict[str, Any]
    resultados: list[dict[str, Any]]
    inicio: float

    resultados = []
    for indice, promt in enumerate(promts, start=1):
        try:
            inicio = perf_counter()
            print(
                f"Petición {indice}/{len(promts)}: {promt['prompt']}",
                flush=True,
            )
            texto = construir_prompt(funciones, promt["prompt"], nombre_modelo)
            ids = modelo.encode(texto).tolist()[0]
            respuesta = generar_respuesta(
                modelo, ids, tokens, restricciones, verbose,
            )
            llamada = validar_llamada(json.loads(respuesta), funciones)
            resultados.append({"prompt": promt["prompt"], **llamada})
            print(
                f"Completada en {perf_counter() - inicio:.2f} s: {respuesta}",
                flush=True,
            )
        except Exception as error:
            raise ErrorProcesamientoPromts(
                f"Petición {indice}: {error}"
            ) from error
    return resultados


def guardar_resultados(ruta: str, resultados: list[dict[str, Any]]) -> None:
    """Escribe el array completo de forma atómica y crea su directorio."""
    destino: Path
    temporal: Path | None

    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=destino.parent, prefix=".function_calls_",
            delete=False,
        ) as archivo:
            temporal = Path(archivo.name)
            json.dump(resultados, archivo, ensure_ascii=True, allow_nan=False,
                      indent=2)
            archivo.write("\n")
        temporal.replace(destino)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Lee las entradas, genera llamadas y guarda los resultados validados."""
    args: argparse.Namespace
    funciones: list[dict[str, Any]]
    promts: list[dict[str, Any]]
    restricciones: Restricciones
    modelo: Any
    vocabulario: dict[str, int]
    tokens: dict[int, bytes]
    resultados: list[dict[str, Any]]
    inicio: float

    args = leer_argumentos(argv)
    inicio = perf_counter()
    try:
        if Path(args.output).resolve() in (
            Path(args.input).resolve(),
            Path(args.functions_definition).resolve(),
        ):
            raise ValueError("La salida no puede sobrescribir las entradas.")
        funciones = load_functions(args.functions_definition)
        promts = load_prompts(args.input)
        restricciones = preparar_restricciones(funciones)
        resultados = []
        if promts:
            if not funciones:
                raise ValueError(
                    "Hay peticiones pero no hay funciones disponibles."
                )
            print(f"Cargando {args.model}...", flush=True)
            modelo = inicializar_modelo(args.model)
            vocabulario = cargar_vocabulario(modelo)
            tokens = preparar_tokens(modelo, vocabulario)
            verificar_tokens(modelo, tokens)
            resultados = procesar_promts(
                modelo, funciones, promts, tokens, restricciones, args.verbose,
                args.model,
            )
        guardar_resultados(args.output, resultados)
        print(
            f"Guardadas {len(resultados)} llamadas en {args.output}. "
            f"Tiempo total: {perf_counter() - inicio:.2f} s.", flush=True,
        )
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
    except KeyboardInterrupt:
        print("Generación interrumpida.", file=sys.stderr)
        return 130
    except Exception as error:
        print(
            f"No se pudo completar el procesamiento: {error}", file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
