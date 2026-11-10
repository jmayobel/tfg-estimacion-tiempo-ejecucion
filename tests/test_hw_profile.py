"""Unit tests for scripts/hw_profile.py."""

import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import hw_profile  # noqa: E402


class TestPureHelpers:
    """Pure functions that don't touch the system."""

    def test_detect_arch_known(self):
        assert hw_profile.detect_arch("x86_64") == "x86_64"
        assert hw_profile.detect_arch("amd64") == "x86_64"
        assert hw_profile.detect_arch("arm64") == "arm64"
        assert hw_profile.detect_arch("aarch64") == "arm64"

    def test_detect_arch_unknown(self):
        assert hw_profile.detect_arch("riscv64") == "riscv64"
        assert hw_profile.detect_arch("") == "unknown"

    def test_int_helper(self):
        assert hw_profile._int("42") == 42
        assert hw_profile._int(None) is None
        assert hw_profile._int("not-a-number") is None
        assert hw_profile._int("") is None

    def test_kb_helper(self):
        assert hw_profile._kb(2048) == 2
        assert hw_profile._kb(0) == 0
        assert hw_profile._kb(None) is None

    def test_first_match_present(self):
        text = "foo: bar\nmodel name: Intel(R) Core\nfoo: baz"
        assert hw_profile._first_match(text, r"^model name:\s*(.+)$") == "Intel(R) Core"

    def test_first_match_missing(self):
        assert hw_profile._first_match("foo bar", r"^missing:\s*(.+)$") is None

    def test_parse_cache_kb_kib(self):
        assert hw_profile._parse_cache_kb("32 KiB (8 instances)") == 32

    def test_parse_cache_kb_mib(self):
        assert hw_profile._parse_cache_kb("30 MiB (1 instance)") == 30 * 1024

    def test_parse_cache_kb_gib(self):
        assert hw_profile._parse_cache_kb("1 GiB") == 1024 * 1024

    def test_parse_cache_kb_invalid(self):
        assert hw_profile._parse_cache_kb("nonsense") is None
        assert hw_profile._parse_cache_kb("") is None


class TestDetectOs:
    """detect_os branches across platforms."""

    def test_detect_os_returns_required_keys(self):
        info = hw_profile.detect_os()
        assert "os_family" in info
        assert "os_release" in info
        assert "machine" in info
        assert info["os_family"] in {"linux", "wsl2", "macos", "windows", "unknown"} or isinstance(
            info["os_family"], str
        )


class TestRun:
    """The _run shell wrapper handles missing commands and non-zero exits."""

    def test_run_missing_command(self):
        assert hw_profile._run(["this-binary-definitely-does-not-exist-xyz"]) is None

    def test_run_nonzero_exit(self):
        # 'false' returns non-zero on POSIX; on Windows we skip.
        import shutil

        if shutil.which("false") is None:
            return
        assert hw_profile._run(["false"]) is None

    def test_run_success(self):
        import shutil

        if shutil.which("echo") is None:
            return
        assert hw_profile._run(["echo", "hello"]) == "hello"


class TestDetectGpu:
    """detect_gpu falls back to 'none' on hosts without nvidia-smi/Apple GPU."""

    def test_detect_gpu_no_nvidia(self):
        with patch.object(hw_profile, "_run", return_value=None):
            info = hw_profile.detect_gpu()
        assert info["gpu_kind"] == "none"
        assert info["gpu_count"] == 0
        assert info["gpu_vendor"] is None

    def test_detect_gpu_nvidia_present(self):
        nv_output = "NVIDIA GeForce RTX 5070 Ti, 16384, 12.0"
        with patch.object(hw_profile, "_run", return_value=nv_output):
            info = hw_profile.detect_gpu()
        assert info["gpu_kind"] == "discrete"
        assert info["gpu_vendor"] == "nvidia"
        assert info["gpu_name"] == "NVIDIA GeForce RTX 5070 Ti"
        assert info["gpu_memory_gb"] == 16.0
        assert info["gpu_compute_capability"] == "12.0"
        assert info["gpu_count"] == 1


class TestBuildProfile:
    """Full profile assembly should never raise and should expose has_gpu."""

    def test_build_profile_basic(self):
        with patch.object(hw_profile, "_run", return_value=None):
            profile = hw_profile.build_profile()
        assert profile["schema_version"] == 1
        assert "hostname_label" in profile
        assert "arch" in profile
        assert "ram_gb" in profile
        assert profile["has_gpu"] is False  # nvidia-smi mocked away

    def test_build_profile_with_gpu(self):
        nv_output = "Test GPU, 8192, 8.6"
        original = hw_profile._run

        def mock_run(cmd, timeout=5.0):
            if cmd and cmd[0] == "nvidia-smi":
                return nv_output
            return original(cmd, timeout)

        with patch.object(hw_profile, "_run", side_effect=mock_run):
            profile = hw_profile.build_profile()
        assert profile["has_gpu"] is True
        assert profile["gpu_kind"] == "discrete"


class TestDockerRam:
    """On macOS the recorded RAM must be what Docker's VM gives the containers."""

    MAC_OS = {"os_family": "macos", "os_release": "25.5.0", "machine": "arm64"}

    def _mock_run(self, docker_out):
        def mock_run(cmd, timeout=5.0):
            if cmd[:2] == ["docker", "info"]:
                return docker_out
            if cmd[:3] == ["sysctl", "-n", "hw.memsize"]:
                return str(24 * 1024**3)
            return None

        return mock_run

    def test_detect_docker_ram_gb(self):
        with patch.object(hw_profile, "_run", side_effect=self._mock_run(str(8 * 1024**3))):
            assert hw_profile.detect_docker_ram_gb() == 8.0
        with patch.object(hw_profile, "_run", return_value=None):
            assert hw_profile.detect_docker_ram_gb() is None

    def _mac_profile(self, docker_out):
        with ExitStack() as stack:
            stack.enter_context(patch.object(hw_profile, "detect_os", return_value=self.MAC_OS))
            stack.enter_context(patch.object(hw_profile, "detect_cpu_macos", return_value={}))
            stack.enter_context(patch.object(hw_profile.shutil, "which", return_value=None))
            stack.enter_context(
                patch.object(hw_profile, "_run", side_effect=self._mock_run(docker_out))
            )
            return hw_profile.build_profile()

    def test_macos_profile_uses_docker_vm_memory(self):
        profile = self._mac_profile(str(8 * 1024**3))
        assert profile["ram_gb"] == 8.0
        assert profile["ram_host_gb"] == 24.0

    def test_macos_without_docker_keeps_host_ram(self):
        profile = self._mac_profile(None)
        assert profile["ram_gb"] == 24.0
        assert "ram_host_gb" not in profile


class TestMain:
    """The CLI entrypoint writes hw_profile.json into --output-dir."""

    def test_main_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = ["hw_profile.py", "--output-dir", tmpdir]
            with patch.object(sys, "argv", argv):
                with patch.object(hw_profile, "_run", return_value=None):
                    rc = hw_profile.main()
            assert rc == 0
            written = Path(tmpdir) / "hw_profile.json"
            assert written.exists()
            data = json.loads(written.read_text())
            assert data["schema_version"] == 1
            assert "arch" in data
