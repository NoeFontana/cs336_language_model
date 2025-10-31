"""Tests for the core module."""

import pytest

from python_template.core import Calculator, greet


class TestGreet:
    """Test cases for the greet function."""

    def test_default_greeting(self):
        """Test greeting with default message."""
        result = greet("World")
        assert result == "Hello, World!"

    def test_custom_greeting(self):
        """Test greeting with custom message."""
        result = greet("Alice", "Hi")
        assert result == "Hi, Alice!"

    def test_empty_name(self):
        """Test greeting with empty name."""
        result = greet("", "Hello")
        assert result == "Hello, !"

    @pytest.mark.parametrize(
        "name,greeting,expected",
        [
            ("John", "Hello", "Hello, John!"),
            ("Jane", "Hi", "Hi, Jane!"),
            ("Bob", "Hey", "Hey, Bob!"),
            ("", "Greetings", "Greetings, !"),
        ],
    )
    def test_greet_parametrized(self, name, greeting, expected):
        """Test greet function with various inputs."""
        result = greet(name, greeting)
        assert result == expected


class TestCalculator:
    """Test cases for the Calculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator instance for testing."""
        return Calculator()

    def test_add(self, calculator):
        """Test addition operation."""
        assert calculator.add(2, 3) == 5
        assert calculator.add(-1, 1) == 0
        assert calculator.add(0, 0) == 0
        assert calculator.add(2.5, 3.7) == pytest.approx(6.2)

    def test_subtract(self, calculator):
        """Test subtraction operation."""
        assert calculator.subtract(5, 3) == 2
        assert calculator.subtract(0, 5) == -5
        assert calculator.subtract(10, 10) == 0
        assert calculator.subtract(5.5, 2.2) == pytest.approx(3.3)

    def test_multiply(self, calculator):
        """Test multiplication operation."""
        assert calculator.multiply(4, 5) == 20
        assert calculator.multiply(0, 100) == 0
        assert calculator.multiply(-2, 3) == -6
        assert calculator.multiply(2.5, 4) == pytest.approx(10.0)

    def test_divide(self, calculator):
        """Test division operation."""
        assert calculator.divide(10, 2) == 5.0
        assert calculator.divide(7, 2) == 3.5
        assert calculator.divide(-8, 2) == -4.0
        assert calculator.divide(1, 3) == pytest.approx(0.3333, rel=1e-3)

    def test_divide_by_zero(self, calculator):
        """Test division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calculator.divide(10, 0)

    @pytest.mark.parametrize(
        "a,b,expected",
        [
            (1, 1, 2),
            (10, -5, 5),
            (0, 0, 0),
            (100, 200, 300),
        ],
    )
    def test_add_parametrized(self, calculator, a, b, expected):
        """Test addition with parametrized inputs."""
        result = calculator.add(a, b)
        assert result == expected

    def test_calculator_operations_chaining(self, calculator):
        """Test chaining multiple operations."""
        # (10 + 5) * 2 - 3 = 27
        result = calculator.add(10, 5)
        result = calculator.multiply(result, 2)
        result = calculator.subtract(result, 3)
        assert result == 27

    @pytest.mark.slow
    def test_large_numbers(self, calculator):
        """Test calculator with large numbers."""
        large_num = 10**10
        result = calculator.add(large_num, large_num)
        assert result == 2 * large_num
