"""Exceptions identifying the stage at which the program fails."""


class ModelLoadError(Exception):
    """Indicate a failure to initialize the model."""


class VocabularyLoadError(Exception):
    """Indicate a failure to read or validate the vocabulary."""


class PromptProcessingError(Exception):
    """Indicate a failure to prepare or generate a response."""
