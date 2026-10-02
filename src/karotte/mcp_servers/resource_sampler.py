"""Resource sampler for collecting CPU and memory metrics during tool execution."""

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from karotte.confinement import Sandbox, current_sandbox
from karotte.schemas.transcript import ResourceMetrics, ResourceSample
from loguru import logger

# cgroup v2 paths for container-level metrics
CGROUP_CPU_STAT = Path("/sys/fs/cgroup/cpu.stat")
CGROUP_MEMORY_CURRENT = Path("/sys/fs/cgroup/memory.current")
# The v2 root (a VM guest, which has no cgroup namespace) has no
# memory.current; its memory.stat has the same charges broken down.
CGROUP_MEMORY_STAT = Path("/sys/fs/cgroup/memory.stat")
_ROOT_MEMORY_STAT_KEYS = ("anon", "file", "kernel", "sock")

# cgroup v1 fallbacks for hybrid layouts (seen in Firecracker VMs), where the
# v2 hierarchy carries no controllers and the numbers live on v1. cpuacct
# reports nanoseconds; the split and combined mount spellings both occur.
CGROUP_V1_CPUACCT_USAGE = (
    Path("/sys/fs/cgroup/cpuacct/cpuacct.usage"),
    Path("/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage"),
)
CGROUP_V1_MEMORY_USAGE = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")


def _trust_cgroup_v1() -> bool:
    """gVisor serves a readable cgroup v1 facade whose numbers mean nothing."""
    return current_sandbox() is not Sandbox.GVISOR


def _read_cgroup_cpu_usage_usec() -> int | None:
    """Cumulative CPU usage in microseconds, from cgroup v2 or v1."""
    try:
        if CGROUP_CPU_STAT.exists():
            for line in CGROUP_CPU_STAT.read_text().splitlines():
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    if not _trust_cgroup_v1():
        return None
    for path in CGROUP_V1_CPUACCT_USAGE:
        try:
            return int(path.read_text().strip()) // 1000
        except (OSError, ValueError):
            continue
    return None


def _read_cgroup_memory_bytes() -> int | None:
    """Current memory usage in bytes, from cgroup v2 (a group's
    memory.current, or the root's memory.stat) or v1."""
    try:
        if CGROUP_MEMORY_CURRENT.exists():
            return int(CGROUP_MEMORY_CURRENT.read_text().strip())
    except (OSError, ValueError):
        pass
    if (root := _read_root_memory_stat_bytes()) is not None:
        return root
    if not _trust_cgroup_v1():
        return None
    try:
        return int(CGROUP_V1_MEMORY_USAGE.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_root_memory_stat_bytes() -> int | None:
    """What the v2 root's memory.stat says is charged: anonymous, page cache,
    kernel and socket memory, the parts memory.current counts elsewhere."""
    try:
        lines = CGROUP_MEMORY_STAT.read_text().splitlines()
    except OSError:
        return None
    stat: dict[str, int] = {}
    for line in lines:
        key, _, value = line.partition(" ")
        if key in _ROOT_MEMORY_STAT_KEYS and value.strip().isdigit():
            stat[key] = int(value)
    if "anon" not in stat or "file" not in stat:
        return None
    return sum(stat.values())


_fallback_logged = False


def _log_psutil_fallback() -> None:
    """Once per process: samplers start on every tool call."""
    global _fallback_logged
    if not _fallback_logged:
        _fallback_logged = True
        logger.info(
            "No cgroup CPU and memory numbers here; resource samples are system-wide (psutil)"
        )


@dataclass
class ResourceSampler:
    """Samples resource usage during command execution.

    Uses cgroup metrics when available (containerized environments),
    falling back to system-wide psutil metrics otherwise.

    Can be used as an async context manager:
        async with ResourceSampler() as sampler:
            # ... run command ...
        metrics = sampler.metrics

    Or with explicit start/stop:
        sampler = ResourceSampler()
        await sampler.start()
        # ... run command ...
        metrics = await sampler.stop()
    """

    sample_interval_s: float = 0.2
    """Sampling interval in seconds (default 200ms)."""

    max_samples: int = 216000  # store up to 12 hours of samples
    """Maximum number of samples to retain."""

    _samples: list[ResourceSample] = field(default_factory=list)
    _task: asyncio.Task[None] | None = None
    _start_time: float = 0.0
    _running: bool = False
    _metrics: ResourceMetrics | None = None
    _use_cgroup: bool = False
    _prev_cpu_usec: int = 0
    _prev_cpu_time: float = 0.0
    _num_cpus: int = field(default_factory=lambda: os.cpu_count() or 1)

    @property
    def metrics(self) -> ResourceMetrics:
        """Get the collected metrics. Available after stop() or exiting context."""
        return self._metrics or ResourceMetrics()

    async def __aenter__(self) -> "ResourceSampler":
        """Start sampling when entering context."""
        await self.start()
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        """Stop sampling when exiting context."""
        self._metrics = await self.stop()

    async def start(self) -> None:
        """Start sampling resources."""
        if self._running:
            return

        self._start_time = time.time()
        self._samples = []
        self._running = True

        # Detect if cgroup metrics are available
        cpu_usec = _read_cgroup_cpu_usage_usec()
        mem_bytes = _read_cgroup_memory_bytes()
        self._use_cgroup = cpu_usec is not None and mem_bytes is not None

        if self._use_cgroup and cpu_usec is not None:
            # Initialize baseline for CPU percentage calculation
            self._prev_cpu_usec = cpu_usec
            self._prev_cpu_time = time.time()
            logger.debug("Using cgroup metrics for resource sampling")
        else:
            # Prime psutil CPU measurement - cpu_percent() stores internal state
            # and computes percentage since last call. First call establishes baseline.
            psutil.cpu_percent()
            _log_psutil_fallback()

        self._task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> ResourceMetrics:
        """Stop sampling and return aggregated metrics."""
        if not self._running:
            return ResourceMetrics()

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        return self._aggregate()

    async def _sample_loop(self) -> None:
        """Continuously sample resources until stopped."""
        while self._running:
            try:
                sample = self._take_sample()
                self._samples.append(sample)

                # Trim samples if we exceed max
                if len(self._samples) > self.max_samples:
                    self._samples = self._samples[-self.max_samples :]

                await asyncio.sleep(self.sample_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Error sampling resources: {e}")
                await asyncio.sleep(self.sample_interval_s)

    def _take_sample(self) -> ResourceSample:
        """Take a resource sample using cgroup or system-wide metrics."""
        timestamp_ms = int((time.time() - self._start_time) * 1000)

        if self._use_cgroup:
            cpu, mem = self._sample_cgroup()
        else:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().used / (1024 * 1024)  # Convert to MB

        return ResourceSample(
            timestamp_ms=timestamp_ms,
            cpu_percent=cpu,
            memory_mb=mem,
        )

    def _sample_cgroup(self) -> tuple[float, float]:
        """Sample CPU and memory from the cgroup readers.

        Returns (cpu_percent, memory_mb).
        CPU percentage is calculated from delta of cumulative CPU time.
        """
        now = time.time()
        cpu_usec = _read_cgroup_cpu_usage_usec() or self._prev_cpu_usec
        mem_bytes = _read_cgroup_memory_bytes() or 0

        # Calculate CPU percentage from delta
        delta_usec = cpu_usec - self._prev_cpu_usec
        delta_time = now - self._prev_cpu_time

        if delta_time > 0:
            # Convert microseconds to seconds, divide by wall time, multiply by 100
            # Divide by num_cpus to normalize to 0-100% range
            cpu_percent = (delta_usec / 1_000_000) / delta_time * 100 / self._num_cpus
        else:
            cpu_percent = 0.0

        # Update baseline for next sample
        self._prev_cpu_usec = cpu_usec
        self._prev_cpu_time = now

        # Convert bytes to MB
        memory_mb = mem_bytes / (1024 * 1024)

        return cpu_percent, memory_mb

    def _aggregate(self) -> ResourceMetrics:
        """Aggregate samples into summary metrics."""
        if not self._samples:
            return ResourceMetrics()

        cpus = [s.cpu_percent for s in self._samples]
        mems = [s.memory_mb for s in self._samples]

        return ResourceMetrics(
            samples=self._samples[-self.max_samples :],
            peak_cpu_percent=max(cpus) if cpus else 0.0,
            avg_cpu_percent=sum(cpus) / len(cpus) if cpus else 0.0,
            peak_memory_mb=max(mems) if mems else 0.0,
            avg_memory_mb=sum(mems) / len(mems) if mems else 0.0,
        )
