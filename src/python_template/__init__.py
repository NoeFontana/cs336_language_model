"""Python Template package.

A modern Python project template with best practices and tooling.
"""

__version__ = "0.1.0"
__author__ = "Noé Fontana"
__email__ = "noe.fontana.pro@gmail.com"

from .core import Calculator, greet

__all__ = ["Calculator", "greet", "__version__"]
