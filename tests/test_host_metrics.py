"""
Unit tests for host_metrics.py
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from host_metrics import (  # noqa: E402
    run_command_safe,
    extract_metric,
    get_total_memory_mb,
    get_gpu_info,
    BYTES_TO_MB,
    DEFAULT_TOTAL_MEMORY_MB,
)


class TestRunCommandSafe:
    """Tests for run_command_safe helper function."""

    def test_successful_command(self):
        """Test successful command execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="  output text  \n")
            result = run_command_safe(["echo", "test"])
            assert result == "output text"

    def test_command_timeout(self):
        """Test command timeout handling."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            result = run_command_safe(["sleep", "10"], timeout=1)
            assert result is None

    def test_command_not_found(self):
        """Test command not found handling."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = run_command_safe(["nonexistent_command"])
            assert result is None

    def test_non_zero_exit_code(self):
        """Test non-zero exit code handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="error")
            result = run_command_safe(["false"])
            assert result is None


class TestExtractMetric:
    """Tests for extract_metric helper function."""

    def test_first_pattern_matches(self):
        """Test extraction when first pattern matches."""
        text = "GPU utilization: 75.5%"
        patterns = [r"GPU\s+utilization\s*:\s*([\d.]+)", r"GPU\s+usage\s*:\s*([\d.]+)"]
        result = extract_metric(text, patterns)
        assert result == 75.5

    def test_second_pattern_matches(self):
        """Test extraction when second pattern matches."""
        text = "GPU usage: 82.3%"
        patterns = [r"GPU\s+utilization\s*:\s*([\d.]+)", r"GPU\s+usage\s*:\s*([\d.]+)"]
        result = extract_metric(text, patterns)
        assert result == 82.3

    def test_no_pattern_matches(self):
        """Test when no pattern matches."""
        text = "CPU usage: 50%"
        patterns = [r"GPU\s+utilization\s*:\s*([\d.]+)", r"GPU\s+usage\s*:\s*([\d.]+)"]
        result = extract_metric(text, patterns)
        assert result is None

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        text = "gpu UTILIZATION: 65.0%"
        patterns = [r"GPU\s+utilization\s*:\s*([\d.]+)"]
        result = extract_metric(text, patterns)
        assert result == 65.0

    def test_custom_parse_function(self):
        """Test custom parse function."""
        text = "Count: 42"
        patterns = [r"Count\s*:\s*(\d+)"]
        result = extract_metric(text, patterns, parse_func=int)
        assert result == 42
        assert isinstance(result, int)

    def test_invalid_value_continues(self):
        """Test that invalid values cause pattern to continue."""
        text = "Value: invalid, Other: 123"
        patterns = [
            r"Value\s*:\s*(\w+)",  # This will match "invalid" which can't be parsed as float
            r"Other\s*:\s*(\d+)",
        ]
        result = extract_metric(text, patterns)
        assert result == 123.0


class TestGetTotalMemoryMb:
    """Tests for get_total_memory_mb function."""

    def test_direct_sysctl_success(self):
        """Test successful memory detection via direct sysctl."""
        with patch("host_metrics.run_command_safe") as mock_cmd:
            # First call returns memory in bytes
            mock_cmd.return_value = "25769803776"
            result = get_total_memory_mb()
            assert result == 25769803776 / BYTES_TO_MB

    def test_key_value_sysctl_fallback(self):
        """Test fallback to key-value format parsing."""
        with patch("host_metrics.run_command_safe") as mock_cmd:
            # First two calls return None (direct queries fail)
            # Third call returns key-value format
            mock_cmd.side_effect = [None, None, "hw.memsize: 17179869184"]
            result = get_total_memory_mb()
            assert result == 17179869184 / BYTES_TO_MB

    def test_all_methods_fail_returns_default(self):
        """Test default value when all detection methods fail."""
        with patch("host_metrics.run_command_safe", return_value=None):
            result = get_total_memory_mb()
            assert result == DEFAULT_TOTAL_MEMORY_MB


class TestGetGpuInfo:
    """Tests for get_gpu_info function."""

    def test_successful_gpu_detection(self):
        """Test successful GPU name detection."""
        gpu_json = """{
            "SPDisplaysDataType": [{
                "sppci_model": "Apple M4 Pro"
            }]
        }"""
        with patch("host_metrics.run_command_safe", return_value=gpu_json):
            result = get_gpu_info()
            assert result == "Apple M4 Pro"

    def test_invalid_json_returns_default(self):
        """Test default value when JSON is invalid."""
        with patch("host_metrics.run_command_safe", return_value="invalid json"):
            result = get_gpu_info()
            assert result == "Apple GPU"

    def test_command_failure_returns_default(self):
        """Test default value when command fails."""
        with patch("host_metrics.run_command_safe", return_value=None):
            result = get_gpu_info()
            assert result == "Apple GPU"

    def test_missing_gpu_model_returns_default(self):
        """Test default when GPU model field is missing."""
        gpu_json = """{
            "SPDisplaysDataType": [{
                "other_field": "value"
            }]
        }"""
        with patch("host_metrics.run_command_safe", return_value=gpu_json):
            result = get_gpu_info()
            assert result == "Apple GPU"

    def test_empty_displays_array_returns_default(self):
        """Test default when displays array is empty."""
        gpu_json = """{
            "SPDisplaysDataType": []
        }"""
        with patch("host_metrics.run_command_safe", return_value=gpu_json):
            result = get_gpu_info()
            assert result == "Apple GPU"
