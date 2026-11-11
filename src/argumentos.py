"""Lectura de las rutas indicadas en la línea de comandos."""

from argparse import ArgumentParser, Namespace


def leer_argumentos() -> Namespace:
    """Lee las rutas de entrada y salida, con valores predeterminados."""
    parser: ArgumentParser

    parser = ArgumentParser(
        description="Traduce peticiones a llamadas de funciones."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Ruta del JSON con las definiciones de funciones.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Ruta del JSON con las peticiones.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Ruta donde guardar los resultados.",
    )
    return parser.parse_args()
