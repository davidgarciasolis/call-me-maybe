*This project has been created as part of the 42 curriculum by davgarc4.*

# Call Me Maybe

## Description

Call Me Maybe translates natural-language requests into function calls using
`Qwen/Qwen3-0.6B` and the supplied `llm_sdk`. It selects a function and extracts
its arguments; it never executes the function.
`google/gemma-3-270m-it` is also available through `--model`.
`openai-community/gpt2` is available for experiments using a plain-text
completion prompt rather than a chat template.

Constrained decoding masks invalid token logits with negative infinity before
choosing the next token. The rules come from the input definitions, so the
program also works with different function names and parameter sets.

## Instructions

Python 3.10 or later and `uv` are required. From the repository root:

```sh
uv sync
uv run python -m src
```

The first run downloads Qwen unless its files are already cached. A CPU run
needs enough memory for the model, its full context and its logits. The SDK
uses float32 on CPU. Avoid running several model instances at once.

Default paths:

- Functions: `data/input/functions_definition.json`
- Requests: `data/input/function_calling_tests.json`
- Results: `data/output/function_calling_results.json`

Custom paths and detailed token progress:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/custom_results.json \
  --verbose
```

Run `uv run python -m src --help` to list the options. Relative paths are
resolved from the working directory. Results are ignored by Git.

To use Gemma instead of the default Qwen model:

```sh
uv run python -m src --model google/gemma-3-270m-it
# Or with the Makefile environment:
make run ARGS="--model google/gemma-3-270m-it"
```

Gemma requires accepting its license on Hugging Face and logging in with
`hf auth login` using an authorized account. If `HF_HOME` points to a separate
cache without a token, the program can reuse the default Hugging Face login.
Explicit `HF_TOKEN` and `HF_TOKEN_PATH` settings take precedence.

To try GPT-2:

```sh
make run ARGS="--model openai-community/gpt2 --verbose"
```

GPT-2 is a base text-completion model, so valid constrained JSON does not
establish correct function selection or argument extraction.

### Makefile

```sh
make install
make run
make run THREADS=4 ARGS="--output data/output/custom_results.json"
make debug
make lint
make clean
make info
```

The Makefile keeps environments, Hugging Face files, uv caches and temporary
files under `~/sgoinfre`. Override this with `SGOINFRE=/another/path` or set
`VENV_DIR` explicitly. `PYTHON` defaults to `python3.10`. Direct CLI execution
uses the environment selected by uv instead. Both paths default to two CPU
threads; `OMP_NUM_THREADS` and `MKL_NUM_THREADS` can be set before direct CLI
execution. `make clean` removes project caches without deleting model files.

## Input and output

The functions file is a JSON array. Each function has `name`, `description`,
`parameters` and `returns`. Parameters are objects with a `type` field:

```json
[
  {
    "name": "add",
    "description": "Add two numbers.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

The requests file contains objects with a string `prompt`:

```json
[{"prompt": "What is the sum of 2 and 3?"}]
```

The result is one array, in input order:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "add",
    "parameters": {"a": 2, "b": 3}
  }
]
```

Supported parameter types are `string`, `number`, `integer`, `boolean` and
`null`. All declared parameters are required. Arrays, objects as arguments,
optional arguments and nested schemas are not supported. Return types are
metadata: no return value is generated.

An empty requests array produces `[]` without loading Qwen. Requests with no
available functions produce a clear error. Invalid JSON, duplicate keys,
unknown argument types and duplicate function names are rejected before
model loading. Empty strings and empty parameter objects are supported.

Output directories are created automatically. The complete result is written
to a temporary file and atomically replaces the destination after successful
generation and validation. Failures leave any previous output intact. Output
paths cannot be the input paths. Errors go to stderr and return status 1;
Ctrl+C returns 130. Argparse reports invalid options with status 2.

## Algorithm explanation

1. Validate both input files and prepare the grammar stages once.
2. Load the selected model once through the SDK's public interface.
3. Read the vocabulary from `tokenizer.json`. For Qwen, invert the ByteLevel
   mapping. For Gemma, replace its space marker and decode byte-fallback
   tokens into bytes. Exclude added tokens, including special tokens, and
   check the mapping against SDK encode/decode.
4. Build a chat prompt for the selected model with the function definitions
   and request. Qwen starts after an empty thinking block; Gemma uses its
   user/model turn markers.
5. Request next-token logits for the current context.
6. For each candidate, decode its bytes with any pending UTF-8 bytes and check
   the resulting response prefix against the grammar. Invalid bytes, unknown
   IDs, special tokens and non-finite logits are blocked. A token may span
   several JSON fields, so its complete content must pass validation.
7. Set every invalid candidate's logit to `-inf`, then choose the largest
   remaining finite logit and append its ID to the model context.
8. Stop only when the call is complete with no UTF-8 bytes pending. Compare
   with SDK decode and validate the complete object against the definitions.
9. Add the original prompt and save the complete results array.

### Design decisions

The grammar requires compact JSON, `name` before `parameters`, and parameters
in definition order. It forbids missing or additional fields. It handles
quoted separators, escaped quotes, control-character escapes and `\uXXXX`
escapes. Numbers are restricted to integers and decimals, without exponents,
as a deliberate project choice. Final validation rejects non-finite numbers.

Grammar validity does not imply that the model chose the correct function or
arguments. Function selection and extraction always come from model logits;
there are no keyword rules or precomputed answers.

The grammar logic is in `src/restricciones.py`; token bytes and logit masking
are in `src/decodificacion.py`. The entry point coordinates loading, generation
and output. No SDK implementation or private attribute is modified or used.

The decoder caches UTF-8 token conversions and the validity of first
characters during each step. Inside an open string without an unfinished
escape it accepts ordinary text directly; tokens containing structural or
escape characters still undergo full validation.

There is deliberately no token limit. A model that keeps extending a string
or number may continue indefinitely; use Ctrl+C to interrupt it. Schema
constraints prevent invalid choices but cannot force a model to finish or
resolve an ambiguous request correctly.

## Challenges faced

A vocabulary token is not literal text, and decoding isolated UTF-8 fragments
can replace incomplete characters. The byte mapping and incremental strict
UTF-8 decoder preserve those fragments until their remaining bytes arrive.

Parsing only completed JSON cannot constrain generation. Separate checks for
valid prefixes and completed values allow incomplete numbers and strings
without permitting their premature closure.

The SDK recomputes the complete context for each token and exposes no inference
cache. Grammar stages and token conversions are reused on the application
side to reduce overhead while keeping the supplied SDK unchanged.

## Testing strategy

Development checks use temporary unittest scripts outside the submission:

- Every prefix of complete calls, all supported types and parameterless calls.
- Escapes, Unicode, delimiters inside strings, wrong keys and trailing text.
- Higher-scoring invalid candidates, special or missing token IDs, non-finite
  logits and UTF-8 split between tokens.
- Comparison of optimized masking with complete prefix validation using real
  vocabulary samples.
- Mocked generation, accumulated contexts, malformed or missing files, empty
  requests and atomic output failure handling.
- Real Qwen inference, followed by independent schema and argument checks.

`make lint` runs the subject's flake8 and mypy commands. Configuration excludes
only the unmodified supplied SDK and virtual environment; all application
modules are checked. Tests are kept outside the delivered repository as
requested by the subject.

## Performance analysis

A real GPT-2 test with the same five functions and sum request correctly
produced `fn_add_numbers` with `a: 2` and `b: 3` and saved valid JSON.
Generation took 122.70 seconds for 23 tokens; total runtime was 137.48 seconds
with two CPU threads. This is one successful example, not an accuracy study.

One real Gemma test with the five sample function definitions correctly
selected `fn_add_numbers` with arguments `a: 2` and `b: 3`, validated the JSON
and saved the result. On this Linux CPU environment with two threads and
7.2 GiB RAM, generation took 197.98 seconds for 19 tokens; total runtime was
208.24 seconds. This single test does not establish overall accuracy or the
five-minute target for the full set. A three-request trial was interrupted
because of the slow generation.

Broader real-model measurements remain pending. The targets are 100%
parseable and schema-compliant
results, at least 90% correct function/argument extraction, and less than five
minutes for the evaluation prompts. These are separate measurements; passing
unit checks alone does not establish them.

## Resources

- The supplied subject, version 1.5, particularly sections V.3–V.6.
- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Python codecs documentation](https://docs.python.org/3/library/codecs.html)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Hugging Face ByteLevel documentation](https://huggingface.co/docs/tokenizers/api/decoders#tokenizers.decoders.ByteLevel)
- The supplied `llm_sdk/llm_sdk/__init__.py` public methods.

AI was used to review the subject and implementation, explain prefix checking,
help implement and refactor the grammar, prepare token bytes, integrate logit
masking, CLI arguments and output handling, and draft documentation and
verification scripts. The author must review, understand and be able to
explain these parts during peer evaluation.
