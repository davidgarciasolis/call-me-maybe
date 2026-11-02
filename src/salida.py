"""Escritura del archivo JSON con las llamadas generadas."""

import json
from pathlib import Path
from typing import Any, TextIO


def guardar_resultados(
    ruta: str,
    resultados: list[dict[str, Any]],
) -> None:
    """Guarda el array JSON, creando los directorios necesarios.

    Sobrescribe el archivo si existe. Propaga los errores de acceso y
    serialización para que el llamador pueda mostrar el motivo del fallo.
    """
    destino: Path
    archivo: TextIO

    destino = Path(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)

    with open(destino, "w") as archivo:
        json.dump(
            resultados,
            archivo,
            ensure_ascii=False,
            indent=4,
            allow_nan=False,
        )
