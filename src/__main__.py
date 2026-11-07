import json
import sys
from time import perf_counter
from typing import Any

from .decodificacion import enmascarar_logits, preparar_tokens
from .error import (
    ErrorCargaModelo,
    ErrorCargaVocabulario,
    ErrorProcesamientoPromts,
)
from .modelo import cargar_vocabulario, inicializar_modelo
from .parseo import load_functions, load_prompts, read_json
from .prompts import construir_prompt
from .restricciones import validar_llamada
from .salida import guardar_resultados


def generar_respuesta(
    modelo: Any,
    ids: list[int],
    funciones: list[dict[str, Any]],
    tokens_permitidos: dict[int, bytes],
) -> str:
    """Genera con máscara y termina cuando json.loads() obtiene un objeto."""
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
    cantidad: int

    paso = 0
    ids_generados = []
    resultado = None
    print("Generando con restricciones...", flush=True)

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

        inicio = perf_counter()
        cantidad = enmascarar_logits(
            logits, ids_generados, tokens_permitidos, funciones
        )
        duracion = perf_counter() - inicio
        print(
            f"Restricciones aplicadas en {duracion:.2f} s. "
            f"Tokens permitidos: {cantidad}.",
            flush=True,
        )
        if cantidad == 0:
            raise ValueError("No quedan tokens permitidos.")

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
    respuesta = b"".join(
        tokens_permitidos[token] for token in ids_generados
    ).decode("utf-8")
    if respuesta != modelo.decode(ids_generados):
        raise ValueError("La respuesta no coincide con la decodificación SDK.")
    return respuesta


def procesar_promts(
    modelo: Any,
    funciones: list[dict[str, Any]],
    promts: list[dict[str, Any]],
    tokens_permitidos: dict[int, bytes],
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
    error: Exception

    resultados = []
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

            respuesta = generar_respuesta(
                modelo, ids, funciones, tokens_permitidos
            )
            llamada = validar_llamada(json.loads(respuesta), funciones)
            resultado = {
                "prompt": promt["prompt"],
                "name": llamada["name"],
                "parameters": llamada["parameters"],
            }
            resultados.append(resultado)
            print("Resultado:")
            print(json.dumps(resultado, ensure_ascii=False, indent=4))
        except Exception as error:
            raise ErrorProcesamientoPromts(
                f"Petición {indice}: {error}"
            ) from error

    return resultados


def main() -> int:
    """Lee las entradas y guarda las llamadas generadas con restricciones."""
    funciones: list[dict[str, Any]]
    promts: list[dict[str, Any]]
    modelo: Any
    vocabulario: dict[str, int]
    tokenizador: Any
    tokens_permitidos: dict[int, bytes]
    resultados: list[dict[str, Any]]
    inicio: float
    duracion: float
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
        tokenizador = read_json(modelo.get_path_to_tokenizer_file())
        if not isinstance(tokenizador, dict):
            raise ValueError("tokenizer.json debe contener un objeto.")
        tokens_permitidos = preparar_tokens(vocabulario, tokenizador)
        print(f"Modelo preparado. Vocabulario: {len(vocabulario)} tokens.")

        inicio = perf_counter()
        resultados = procesar_promts(
            modelo, funciones, promts, tokens_permitidos
        )
        duracion = perf_counter() - inicio
        guardar_resultados(
            "data/output/function_calling_results.json", resultados
        )
        print(
            f"Tiempo total de generación de todas las respuestas: "
            f"{duracion:.2f} s.",
            flush=True,
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
        print(f"No se pudo leer o guardar un archivo: {error}", file=sys.stderr)
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
