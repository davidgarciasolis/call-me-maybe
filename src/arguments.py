"""Read paths provided on the command line."""

from argparse import ArgumentParser, Namespace


def parse_arguments() -> Namespace:
    """Read input and output paths, using default values."""
    parser: ArgumentParser

    parser = ArgumentParser(
        description="Translate requests into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing requests.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path where results will be saved.",
    )
    return parser.parse_args()
