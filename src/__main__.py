import json
import sys
from argparse import Namespace
from time import perf_counter
from typing import Any

from .arguments import parse_arguments
from .decoding import prepare_tokens, select_token
from .error import (
    ModelLoadError,
    VocabularyLoadError,
    PromptProcessingError,
)
from .model import load_vocabulary, initialize_model
from .parsing import load_functions, load_prompts
from .prompts import build_prompt
from .output import save_results


def generate_response(
    model: Any,
    ids: list[int],
    functions: list[dict[str, Any]],
    decoded_tokens: dict[int, str],
) -> str:
    """Choose valid tokens until json.loads() returns an object."""
    logits: list[float]
    generated_ids: list[int]
    next_id: int
    response: str
    result: Any

    generated_ids = []
    response = ""
    result = None

    while not isinstance(result, dict):
        logits = model.get_logits_from_input_ids(ids)

        if not logits:
            raise ValueError("The model returned no logits.")

        next_id = select_token(
            logits, response, decoded_tokens, functions
        )
        ids.append(next_id)
        generated_ids.append(next_id)

        response += decoded_tokens[next_id]

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            continue

    if model.decode(generated_ids) != response:
        raise ValueError(
            "The concatenated text does not match the SDK output."
        )
    return response


def process_prompts(
    model: Any,
    functions: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    decoded_tokens: dict[int, str],
) -> list[dict[str, Any]]:
    """Generate and return results, reusing the same model."""
    prompt: dict[str, Any]
    index: int
    text: str
    tokens: Any
    id_lists: list[list[int]]
    ids: list[int]
    response: str
    call: dict[str, Any]
    result: dict[str, Any]
    results: list[dict[str, Any]]
    start: float
    duration: float

    results = []
    for index, prompt in enumerate(prompts, start=1):
        try:
            print(
                f"Processing request {index}/{len(prompts)}...",
                flush=True,
            )
            start = perf_counter()
            text = build_prompt(functions, prompt["prompt"])

            tokens = model.encode(text)
            id_lists = tokens.tolist()
            ids = id_lists[0]

            response = generate_response(
                model, ids, functions, decoded_tokens
            )
            call = json.loads(response)
            result = {
                "prompt": prompt["prompt"],
                "name": call["name"],
                "parameters": call["parameters"],
            }
            results.append(result)
            duration = perf_counter() - start
            print(
                f"Request {index}/{len(prompts)} completed "
                f"in {duration:.2f} s.",
                flush=True,
            )
        except Exception as error:
            raise PromptProcessingError(
                f"Request {index}: {error}"
            ) from error

    return results


def main() -> int:
    """Read inputs and save calls generated with constraints."""
    arguments: Namespace
    functions: list[dict[str, Any]]
    prompts: list[dict[str, Any]]
    model: Any
    vocabulary: dict[str, int]
    decoded_tokens: dict[int, str]
    results: list[dict[str, Any]]
    start: float
    duration: float

    try:
        start = perf_counter()
        arguments = parse_arguments()
        functions = load_functions(arguments.functions_definition)
        prompts = load_prompts(arguments.input)

        if prompts and not functions:
            raise ValueError(
                "There are requests, but no functions are available."
            )

        print("Loading the model.", flush=True)
        model = initialize_model()
        vocabulary = load_vocabulary(model)
        decoded_tokens = prepare_tokens(model, vocabulary)
        print("Model ready. Encoding on the CPU", flush=True)

        results = process_prompts(
            model, functions, prompts, decoded_tokens
        )
        save_results(arguments.output, results)
        duration = perf_counter() - start
        print(f"Results saved to {arguments.output}", flush=True)
        print(
            f"Total time: {duration:.2f} s.",
            flush=True,
        )
    except ModelLoadError as error:
        print(f"Could not load the model: {error}", file=sys.stderr)
        return 1
    except VocabularyLoadError as error:
        print(
            f"Could not load the vocabulary: {error}", file=sys.stderr
        )
        return 1
    except PromptProcessingError as error:
        print(
            f"Could not process the prompts: {error}", file=sys.stderr
        )
        return 1
    except OSError as error:
        print(
            f"Could not read or save a file: {error}", file=sys.stderr
        )
        return 1
    except ValueError as error:
        print(f"Invalid content or value: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Execution interrupted by the user.", file=sys.stderr)
        return 130
    except Exception as error:
        print(
            f"Could not finish loading or process the prompts: {error}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
