import asyncio
from pathlib import Path

import pytest
from loguru import logger

from karotte.mcp_servers import resource_sampler
from karotte.mcp_servers.resource_sampler import (
    ResourceSampler,
    _read_cgroup_cpu_usage_usec,  # pyright: ignore[reportPrivateUsage]
    _read_cgroup_memory_bytes,  # pyright: ignore[reportPrivateUsage]
)
from karotte.schemas.transcript import ResourceMetrics


class TestResourceSampler:
    @pytest.mark.asyncio
    async def test_context_manager_returns_metrics(self):
        """Test that async context manager collects and returns metrics."""
        async with ResourceSampler(sample_interval_s=0.05) as sampler:
            await asyncio.sleep(0.15)  # Allow a few samples

        metrics = sampler.metrics
        assert isinstance(metrics, ResourceMetrics)
        assert len(metrics.samples) > 0

    @pytest.mark.asyncio
    async def test_metrics_have_expected_fields(self):
        """Test that collected metrics have all expected fields populated."""
        async with ResourceSampler(sample_interval_s=0.05) as sampler:
            await asyncio.sleep(0.15)

        metrics = sampler.metrics
        assert metrics.peak_cpu_percent >= 0
        assert metrics.avg_cpu_percent >= 0
        assert metrics.peak_memory_mb >= 0
        assert metrics.avg_memory_mb >= 0

    @pytest.mark.asyncio
    async def test_samples_have_increasing_timestamps(self):
        """Test that sample timestamps increase monotonically."""
        async with ResourceSampler(sample_interval_s=0.05) as sampler:
            await asyncio.sleep(0.2)

        metrics = sampler.metrics
        if len(metrics.samples) > 1:
            for i in range(1, len(metrics.samples)):
                assert (
                    metrics.samples[i].timestamp_ms
                    > metrics.samples[i - 1].timestamp_ms
                )

    @pytest.mark.asyncio
    async def test_max_samples_limit(self):
        """Test that samples are trimmed to max_samples."""
        async with ResourceSampler(sample_interval_s=0.01, max_samples=5) as sampler:
            await asyncio.sleep(0.15)  # Should generate more than 5 samples

        metrics = sampler.metrics
        assert len(metrics.samples) <= 5

    @pytest.mark.asyncio
    async def test_empty_metrics_when_stopped_immediately(self):
        """Test that stopping immediately returns empty/default metrics."""
        sampler = ResourceSampler()
        metrics = await sampler.stop()

        assert isinstance(metrics, ResourceMetrics)
        assert len(metrics.samples) == 0

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test explicit start/stop lifecycle."""
        sampler = ResourceSampler(sample_interval_s=0.05)
        await sampler.start()
        await asyncio.sleep(0.15)
        metrics = await sampler.stop()

        assert isinstance(metrics, ResourceMetrics)
        assert len(metrics.samples) > 0

    @pytest.mark.asyncio
    async def test_multiple_start_calls_are_idempotent(self):
        """Test that calling start() multiple times doesn't cause issues."""
        sampler = ResourceSampler(sample_interval_s=0.05)
        await sampler.start()
        await sampler.start()  # Should be no-op
        await asyncio.sleep(0.1)
        metrics = await sampler.stop()

        assert len(metrics.samples) > 0

    @pytest.mark.asyncio
    async def test_aggregate_calculates_correct_peak(self):
        """Test that peak values are correctly calculated."""
        async with ResourceSampler(sample_interval_s=0.05) as sampler:
            await asyncio.sleep(0.15)

        metrics = sampler.metrics
        if metrics.samples:
            actual_peak_cpu = max(s.cpu_percent for s in metrics.samples)
            actual_peak_mem = max(s.memory_mb for s in metrics.samples)
            assert metrics.peak_cpu_percent == actual_peak_cpu
            assert metrics.peak_memory_mb == actual_peak_mem

    @pytest.mark.asyncio
    async def test_aggregate_calculates_correct_average(self):
        """Test that average values are correctly calculated."""
        async with ResourceSampler(sample_interval_s=0.05) as sampler:
            await asyncio.sleep(0.15)

        metrics = sampler.metrics
        if metrics.samples:
            actual_avg_cpu = sum(s.cpu_percent for s in metrics.samples) / len(
                metrics.samples
            )
            actual_avg_mem = sum(s.memory_mb for s in metrics.samples) / len(
                metrics.samples
            )
            assert abs(metrics.avg_cpu_percent - actual_avg_cpu) < 0.001
            assert abs(metrics.avg_memory_mb - actual_avg_mem) < 0.001


class TestCgroupHelpers:
    def test_read_cgroup_cpu_returns_none_when_file_missing(self):
        """Test that cgroup CPU reader returns None when file doesn't exist."""
        # On most dev machines, cgroup v2 paths won't exist
        result = _read_cgroup_cpu_usage_usec()
        # Result is either None (no cgroup) or an int (in container)
        assert result is None or isinstance(result, int)

    def test_read_cgroup_memory_returns_none_when_file_missing(self):
        """Test that cgroup memory reader returns None when file doesn't exist."""
        result = _read_cgroup_memory_bytes()
        assert result is None or isinstance(result, int)


class TestCgroupV1Fallback:
    """A hybrid layout (seen in Firecracker VMs) mounts cgroup2 with no
    controllers on it; the real numbers live on cgroup v1. Reading only the v2
    paths there silently falls back to system-wide psutil numbers."""

    def _absent_v2(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(resource_sampler, "CGROUP_CPU_STAT", tmp_path / "absent")
        monkeypatch.setattr(
            resource_sampler, "CGROUP_MEMORY_CURRENT", tmp_path / "absent"
        )
        monkeypatch.setattr(resource_sampler, "CGROUP_MEMORY_STAT", tmp_path / "absent")

    def test_cpu_falls_back_to_v1_cpuacct(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """cpuacct reports nanoseconds; the caller expects microseconds."""
        monkeypatch.setenv("KAROTTE_SANDBOX", "firecracker")
        self._absent_v2(monkeypatch, tmp_path)
        usage = tmp_path / "cpuacct.usage"
        usage.write_text("123456789\n")
        monkeypatch.setattr(resource_sampler, "CGROUP_V1_CPUACCT_USAGE", (usage,))

        assert _read_cgroup_cpu_usage_usec() == 123456

    def test_memory_falls_back_to_v1_usage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("KAROTTE_SANDBOX", "firecracker")
        self._absent_v2(monkeypatch, tmp_path)
        usage = tmp_path / "memory.usage_in_bytes"
        usage.write_text("262144\n")
        monkeypatch.setattr(resource_sampler, "CGROUP_V1_MEMORY_USAGE", usage)

        assert _read_cgroup_memory_bytes() == 262144

    def test_the_gvisor_facade_is_not_trusted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """gVisor serves readable v1 files whose numbers mean nothing."""
        monkeypatch.setenv("KAROTTE_SANDBOX", "gvisor")
        self._absent_v2(monkeypatch, tmp_path)
        cpu = tmp_path / "cpuacct.usage"
        cpu.write_text("123456789\n")
        mem = tmp_path / "memory.usage_in_bytes"
        mem.write_text("262144\n")
        monkeypatch.setattr(resource_sampler, "CGROUP_V1_CPUACCT_USAGE", (cpu,))
        monkeypatch.setattr(resource_sampler, "CGROUP_V1_MEMORY_USAGE", mem)

        assert _read_cgroup_cpu_usage_usec() is None
        assert _read_cgroup_memory_bytes() is None

    def test_v2_still_wins_when_it_carries_the_controllers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.setenv("KAROTTE_SANDBOX", "firecracker")
        stat = tmp_path / "cpu.stat"
        stat.write_text("usage_usec 42\nuser_usec 20\n")
        current = tmp_path / "memory.current"
        current.write_text("1024\n")
        monkeypatch.setattr(resource_sampler, "CGROUP_CPU_STAT", stat)
        monkeypatch.setattr(resource_sampler, "CGROUP_MEMORY_CURRENT", current)

        assert _read_cgroup_cpu_usage_usec() == 42
        assert _read_cgroup_memory_bytes() == 1024


ROOT_MEMORY_STAT = """\
anon 1000
file 200
kernel 30
kernel_stack 8
sock 4
shmem 50
file_mapped 70
"""


class TestCgroupV2Root:
    """A VM guest has no cgroup namespace, so /sys/fs/cgroup is the v2 root:
    cpu.stat is there, memory.current isn't, and cgroup v1 is off."""

    @pytest.fixture
    def v2_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        monkeypatch.setenv("KAROTTE_SANDBOX", "vm")
        stat = tmp_path / "cpu.stat"
        stat.write_text("usage_usec 42\nuser_usec 20\n")
        monkeypatch.setattr(resource_sampler, "CGROUP_CPU_STAT", stat)
        monkeypatch.setattr(
            resource_sampler, "CGROUP_MEMORY_CURRENT", tmp_path / "absent"
        )
        monkeypatch.setattr(resource_sampler, "CGROUP_V1_CPUACCT_USAGE", ())
        monkeypatch.setattr(
            resource_sampler, "CGROUP_V1_MEMORY_USAGE", tmp_path / "absent"
        )
        memory_stat = tmp_path / "memory.stat"
        monkeypatch.setattr(resource_sampler, "CGROUP_MEMORY_STAT", memory_stat)
        return memory_stat

    def test_memory_comes_from_the_root_memory_stat(self, v2_root: Path):
        """anon + file + kernel + sock; the finer keys are parts of those."""
        _ = v2_root.write_text(ROOT_MEMORY_STAT)

        assert _read_cgroup_memory_bytes() == 1000 + 200 + 30 + 4

    def test_an_older_kernel_without_the_kernel_key_still_counts(self, v2_root: Path):
        _ = v2_root.write_text("anon 1000\nfile 200\nsock 4\n")

        assert _read_cgroup_memory_bytes() == 1204

    def test_an_unreadable_memory_stat_gives_nothing(self, v2_root: Path):
        _ = v2_root.write_text("garbage\n")

        assert _read_cgroup_memory_bytes() is None

    @pytest.mark.asyncio
    async def test_the_sampler_uses_the_cgroup_numbers(self, v2_root: Path):
        _ = v2_root.write_text(ROOT_MEMORY_STAT)

        async with ResourceSampler(sample_interval_s=0.01) as sampler:
            await asyncio.sleep(0.05)

        assert sampler._use_cgroup  # pyright: ignore[reportPrivateUsage]
        assert sampler.metrics.peak_memory_mb == pytest.approx(1234 / (1024 * 1024))

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("v2_root")
    async def test_the_psutil_fallback_is_logged_once(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(resource_sampler, "_fallback_logged", False)
        messages: list[str] = []
        handler = logger.add(messages.append, level="INFO", format="{message}")
        try:
            for _ in range(3):
                async with ResourceSampler(sample_interval_s=0.01):
                    pass
        finally:
            logger.remove(handler)

        assert len([m for m in messages if "system-wide" in m]) == 1
