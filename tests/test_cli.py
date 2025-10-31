"""Tests for the CLI module."""

import pytest

from python_template.cli import create_parser, main


class TestCLI:
    """Test cases for the CLI functionality."""

    def test_create_parser(self):
        """Test that the parser is created correctly."""
        parser = create_parser()
        assert parser.prog == "python-template"

    def test_greet_command(self, capsys):
        """Test the greet command."""
        result = main(["greet", "World"])
        captured = capsys.readouterr()
        assert result == 0
        assert "Hello, World!" in captured.out

    def test_greet_command_with_custom_greeting(self, capsys):
        """Test the greet command with custom greeting."""
        result = main(["greet", "Alice", "--greeting", "Hi"])
        captured = capsys.readouterr()
        assert result == 0
        assert "Hi, Alice!" in captured.out

    def test_calc_add(self, capsys):
        """Test calculator addition."""
        result = main(["calc", "add", "2", "3"])
        captured = capsys.readouterr()
        assert result == 0
        assert "Result: 5" in captured.out

    def test_calc_subtract(self, capsys):
        """Test calculator subtraction."""
        result = main(["calc", "sub", "10", "4"])
        captured = capsys.readouterr()
        assert result == 0
        assert "Result: 6" in captured.out

    def test_calc_multiply(self, capsys):
        """Test calculator multiplication."""
        result = main(["calc", "mul", "6", "7"])
        captured = capsys.readouterr()
        assert result == 0
        assert "Result: 42" in captured.out

    def test_calc_divide(self, capsys):
        """Test calculator division."""
        result = main(["calc", "div", "15", "3"])
        captured = capsys.readouterr()
        assert result == 0
        assert "Result: 5" in captured.out

    def test_calc_divide_by_zero(self, capsys):
        """Test calculator division by zero."""
        result = main(["calc", "div", "10", "0"])
        captured = capsys.readouterr()
        assert result == 1
        assert "Error: Cannot divide by zero" in captured.err

    def test_no_command(self, capsys):
        """Test behavior when no command is provided."""
        result = main([])
        captured = capsys.readouterr()
        assert result == 1
        assert "usage:" in captured.out

    @pytest.mark.parametrize(
        "operation,a,b,expected",
        [
            ("add", "1", "2", "3"),
            ("sub", "10", "3", "7"),
            ("mul", "4", "5", "20"),
            ("div", "12", "4", "3"),
        ],
    )
    def test_calc_operations_parametrized(self, capsys, operation, a, b, expected):
        """Test calculator operations with parametrized inputs."""
        result = main(["calc", operation, a, b])
        captured = capsys.readouterr()
        assert result == 0
        assert f"Result: {expected}" in captured.out
