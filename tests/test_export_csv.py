"""
Unit tests for export_csv.py
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from export_csv import query_prometheus_range, query_with_fallbacks, get_cpu_clock  # noqa: E402


class TestQueryPrometheusRange:
    """Tests for query_prometheus_range function."""

    def test_successful_query(self, requests_mock):
        """Test successful Prometheus query returns parsed values."""
        requests_mock.get(
            "http://localhost:9090/api/v1/query_range",
            json={
                "status": "success",
                "data": {"result": [{"values": [[1609459200, "42.5"], [1609459202, "43.0"]]}]},
            },
        )

        result = query_prometheus_range(
            "http://localhost:9090", "test_metric", 1609459200, 1609459210, 2
        )

        assert len(result) == 2
        assert result[0] == (1609459200.0, 42.5)
        assert result[1] == (1609459202.0, 43.0)

    def test_timeout(self, requests_mock):
        """Test timeout handling returns empty list."""
        import requests

        requests_mock.get("http://localhost:9090/api/v1/query_range", exc=requests.Timeout)

        result = query_prometheus_range(
            "http://localhost:9090", "test_metric", 1609459200, 1609459210, 2
        )

        assert result == []

    def test_invalid_json(self, requests_mock):
        """Test invalid JSON response handling."""
        requests_mock.get("http://localhost:9090/api/v1/query_range", text="invalid json")

        result = query_prometheus_range(
            "http://localhost:9090", "test_metric", 1609459200, 1609459210, 2
        )

        assert result == []

    def test_non_success_status(self, requests_mock):
        """Test non-success status handling."""
        requests_mock.get(
            "http://localhost:9090/api/v1/query_range",
            json={"status": "error", "error": "query timeout"},
        )

        result = query_prometheus_range(
            "http://localhost:9090", "test_metric", 1609459200, 1609459210, 2
        )

        assert result == []

    def test_empty_results(self, requests_mock):
        """Test empty results handling."""
        requests_mock.get(
            "http://localhost:9090/api/v1/query_range",
            json={"status": "success", "data": {"result": []}},
        )

        result = query_prometheus_range(
            "http://localhost:9090", "test_metric", 1609459200, 1609459210, 2
        )

        assert result == []

    def test_malformed_response(self, requests_mock):
        """Test malformed response structure."""
        requests_mock.get(
            "http://localhost:9090/api/v1/query_range",
            json={"status": "success", "data": "malformed"},
        )

        result = query_prometheus_range(
            "http://localhost:9090", "test_metric", 1609459200, 1609459210, 2
        )

        assert result == []


class TestQueryWithFallbacks:
    """Tests for query_with_fallbacks function."""

    def test_first_query_succeeds(self, requests_mock):
        """Test first query succeeding returns immediately."""
        requests_mock.get(
            "http://localhost:9090/api/v1/query_range",
            json={"status": "success", "data": {"result": [{"values": [[1609459200, "50.0"]]}]}},
        )

        queries = ['metric{{name="{container}"}}', 'metric{{service="{service_key}"}}']

        values, matched_query = query_with_fallbacks(
            "http://localhost:9090",
            queries,
            1609459200,
            1609459210,
            2,
            "test-container",
            "test-service",
        )

        assert len(values) == 1
        assert matched_query == 'metric{name="test-container"}'

    def test_all_queries_fail(self, requests_mock):
        """Test all queries failing returns empty list and None."""
        requests_mock.get(
            "http://localhost:9090/api/v1/query_range",
            json={"status": "success", "data": {"result": []}},
        )

        queries = ['metric{{name="{container}"}}', 'metric{{service="{service_key}"}}']

        values, matched_query = query_with_fallbacks(
            "http://localhost:9090",
            queries,
            1609459200,
            1609459210,
            2,
            "test-container",
            "test-service",
        )

        assert values == []
        assert matched_query is None


class TestGetCpuClock:
    """Tests for get_cpu_clock function."""

    def test_linux_cpuinfo_success(self):
        """Test reading CPU clock from /proc/cpuinfo on Linux."""
        cpuinfo_content = """processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 142
model name	: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
cpu MHz		: 2400.000
"""
        with patch("builtins.open", mock_open(read_data=cpuinfo_content)):
            result = get_cpu_clock()
            assert result == 2400.0

    def test_linux_cpuinfo_not_found(self):
        """Test fallback when /proc/cpuinfo doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="2400000000\n")
                result = get_cpu_clock()
                assert result == 2400.0

    def test_macos_sysctl_success(self):
        """Test reading CPU clock from sysctl on macOS."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="3200000000\n")
                result = get_cpu_clock()
                assert result == 3200.0

    def test_all_methods_fail(self):
        """Test when all detection methods fail."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=1, stdout="")
                result = get_cpu_clock()
                assert result is None


# Fixtures
@pytest.fixture
def requests_mock():
    """Fixture to provide requests-mock for HTTP mocking."""
    import requests_mock as rm

    with rm.Mocker() as m:
        yield m
