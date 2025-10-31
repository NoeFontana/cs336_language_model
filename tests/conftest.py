"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "numbers": [1, 2, 3, 4, 5],
        "text": "Hello, World!",
        "dict": {"key": "value", "number": 42},
    }
