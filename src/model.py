"""Initialize the model and load its vocabulary."""

from llm_sdk import Small_LLM_Model

from .error import ModelLoadError, VocabularyLoadError
from .parsing import read_json


def initialize_model() -> Small_LLM_Model:
    """Load the Qwen model and its tokenizer."""
    model: Small_LLM_Model

    try:
        model = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    except Exception as error:
        raise ModelLoadError(str(error)) from error
    return model


def load_vocabulary(model: Small_LLM_Model) -> dict[str, int]:
    """Read the mapping between model tokens and their IDs."""
    path: str
    vocabulary: dict[str, int]

    try:
        path = model.get_path_to_vocab_file()
        vocabulary = read_json(path)
    except Exception as error:
        raise VocabularyLoadError(str(error)) from error

    return vocabulary
