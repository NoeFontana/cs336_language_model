"""Core functionality for the Python template package."""


def greet(name: str, greeting: str = "Hello") -> str:
    """Greet someone with a custom message.

    Args:
        name: The name of the person to greet.
        greeting: The greeting message to use. Defaults to "Hello".

    Returns:
        A formatted greeting string.

    Examples:
        >>> greet("World")
        'Hello, World!'
        >>> greet("Alice", "Hi")
        'Hi, Alice!'
    """
    return f"{greeting}, {name}!"


class Calculator:
    """A simple calculator class for demonstration purposes."""

    def add(self, a: int | float, b: int | float) -> int | float:
        """Add two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The sum of a and b.

        Examples:
            >>> calc = Calculator()
            >>> calc.add(2, 3)
            5
            >>> calc.add(2.5, 3.7)
            6.2
        """
        return a + b

    def subtract(self, a: int | float, b: int | float) -> int | float:
        """Subtract two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The difference of a and b.

        Examples:
            >>> calc = Calculator()
            >>> calc.subtract(5, 3)
            2
            >>> calc.subtract(5.5, 2.2)
            3.3
        """
        return a - b

    def multiply(self, a: int | float, b: int | float) -> int | float:
        """Multiply two numbers.

        Args:
            a: The first number.
            b: The second number.

        Returns:
            The product of a and b.

        Examples:
            >>> calc = Calculator()
            >>> calc.multiply(4, 5)
            20
            >>> calc.multiply(2.5, 4)
            10.0
        """
        return a * b

    def divide(self, a: int | float, b: int | float) -> float:
        """Divide two numbers.

        Args:
            a: The dividend.
            b: The divisor.

        Returns:
            The quotient of a and b.

        Raises:
            ValueError: If b is zero.

        Examples:
            >>> calc = Calculator()
            >>> calc.divide(10, 2)
            5.0
            >>> calc.divide(7, 2)
            3.5
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
