"""Command-line interface for the python-template package."""

import argparse
import sys

from python_template import Calculator, greet


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Python Template CLI",
        prog="python-template",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Greet command
    greet_parser = subparsers.add_parser("greet", help="Greet someone")
    greet_parser.add_argument("name", help="Name to greet")
    greet_parser.add_argument(
        "--greeting",
        default="Hello",
        help="Greeting message (default: Hello)",
    )

    # Calculator command
    calc_parser = subparsers.add_parser("calc", help="Perform calculations")
    calc_parser.add_argument("operation", choices=["add", "sub", "mul", "div"])
    calc_parser.add_argument("a", type=float, help="First number")
    calc_parser.add_argument("b", type=float, help="Second number")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "greet":
        greeting_result = greet(args.name, args.greeting)
        print(greeting_result)
        return 0

    elif args.command == "calc":
        calc = Calculator()
        try:
            calc_result: float
            if args.operation == "add":
                calc_result = calc.add(args.a, args.b)
            elif args.operation == "sub":
                calc_result = calc.subtract(args.a, args.b)
            elif args.operation == "mul":
                calc_result = calc.multiply(args.a, args.b)
            elif args.operation == "div":
                calc_result = calc.divide(args.a, args.b)
            else:
                print(f"Unknown operation: {args.operation}", file=sys.stderr)
                return 1

            print(f"Result: {calc_result}")
            return 0

        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
