"""
Pytest configuration and shared fixtures.
"""

import pytest
import json
from pathlib import Path


@pytest.fixture
def sample_prometheus_response():
    """Load sample Prometheus response from fixtures."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_prometheus_response.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def mock_prometheus_success():
    """Return a mock successful Prometheus response."""
    return {
        "status": "success",
        "data": {
            "result": [
                {"values": [[1609459200, "42.5"], [1609459202, "43.0"], [1609459204, "44.2"]]}
            ]
        },
    }


@pytest.fixture
def mock_prometheus_error():
    """Return a mock error Prometheus response."""
    return {"status": "error", "error": "query timeout", "errorType": "timeout"}


@pytest.fixture
def mock_prometheus_empty():
    """Return a mock empty Prometheus response."""
    return {"status": "success", "data": {"result": []}}
