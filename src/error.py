"""Excepciones para identificar la etapa en la que falla el programa."""


class ErrorCargaModelo(Exception):
    """Indica un fallo al inicializar el modelo."""


class ErrorCargaVocabulario(Exception):
    """Indica un fallo al leer o validar el vocabulario."""


class ErrorProcesamientoPromts(Exception):
    """Indica un fallo al preparar o generar una respuesta."""
