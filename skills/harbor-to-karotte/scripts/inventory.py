"""Triage a Harbor dataset before porting it to karotte.

Usage: python inventory.py <dataset-dir> [--json out.json]

Finds every Harbor task (a directory with task.toml) under <dataset-dir> and
prints one row per task: hardware, timeouts, verifier mode, base image, and the
features that make a port harder. The verdict column is a first guess for the
triage report, not a decision: read the flagged tasks before you trust it.
"""

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

# Base images whose packages map onto Amazon Linux 2023 with a few dnf lines.
PLAIN_BASES = re.compile(
    r"^(ubuntu|debian|python|buildpack-deps|alpine"
    + r"|ghcr\.io/laude-institute/t-bench/(ubuntu|python)[^ ]*)(:|@|$)"
)


# Flags that mean "read this before porting" but don't by themselves make the
# port harder than usual.
NOTE_FLAGS = ("verifier-image", "conftest", "extra-dirs")

# What a Harbor task folder has; anything else (cheat/, notes/) isn't part of
# the format and may not be on the hub.
STANDARD_DIRS = {"environment", "tests", "solution"}


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _hardware(cfg: dict[str, Any]) -> dict[str, Any]:
    env = cfg.get("environment") or {}
    return {
        "cpus": env.get("cpus"),
        "memory_mb": env.get("memory_mb"),
        "storage_mb": env.get("storage_mb"),
        "gpus": env.get("gpus") or 0,
    }


def inspect(task_dir: Path) -> dict[str, Any]:
    cfg = tomllib.loads(_read(task_dir / "task.toml"))
    task = cfg.get("task") or {}
    verifier = cfg.get("verifier") or {}
    env_dir = task_dir / "environment"
    dockerfile = _read(env_dir / "Dockerfile")
    bases = re.findall(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)", dockerfile, re.M)
    solve = "\n".join(
        _read(p) for p in (task_dir / "solution").rglob("*") if p.is_file()
    )
    # test.sh downloading uv or pytest is Harbor boilerplate; the port replaces it.
    tests = "\n".join(
        _read(p)
        for p in (task_dir / "tests").rglob("*")
        if p.is_file() and p.name != "test.sh"
    )
    hw = _hardware(cfg)
    flags = []
    if hw["gpus"]:
        flags.append(f"gpus={hw['gpus']}")
    if any(
        (env_dir / n).exists()
        for n in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml")
    ):
        flags.append("multi-container")
    if bases and not all(PLAIN_BASES.match(b) for b in bases):
        flags.append("custom-base:" + ",".join(bases))
    if len(bases) > 1:
        flags.append("multi-stage")
    if re.search(r"\b(sudo|apt-get|apt|systemctl|service)\b", solve):
        flags.append("root-in-solution")
    if re.search(r"\b(pip|uv pip|npm|cargo|go) (install|get|add)\b", solve):
        flags.append("installs-in-solution")
    if re.search(r"https?://(?!localhost|127\.0\.0\.1)", tests):
        flags.append("network-in-tests")
    if re.search(
        r"(localhost|127\.0\.0\.1):\d+|\bsocket\b|requests\.(get|post)", tests
    ):
        flags.append("service-check")
    if re.search(r"\b(curl|wget)\b[^\n]*(\|\s*(ba)?sh|https?://)", dockerfile):
        flags.append("build-downloads")
    if (task_dir / "tests" / "Dockerfile").is_file():
        flags.append("verifier-image")
    if any((task_dir / "tests").rglob("conftest.py")):
        flags.append("conftest")
    extra = sorted(
        d.name
        for d in task_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in STANDARD_DIRS
    )
    if extra:
        flags.append("extra-dirs:" + ",".join(extra))
    if not (task_dir / "solution").is_dir() or not solve.strip():
        flags.append("no-solution")
    if not (task_dir / "tests").is_dir():
        flags.append("no-tests")

    # GPU tasks can't run on karotte's VM runtimes yet; no tests means nothing to score.
    if any(f.startswith("gpus") or f == "no-tests" for f in flags):
        verdict = "skip-or-defer"
    elif any(not f.startswith(NOTE_FLAGS) for f in flags):
        verdict = "needs-adaptation"
    else:
        verdict = "portable"
    size = sum(p.stat().st_size for p in task_dir.rglob("*") if p.is_file())
    return {
        "dir": str(task_dir),
        "name": task.get("name") or task_dir.name,
        "authors": [a.get("name") for a in task.get("authors") or [] if a.get("name")],
        "organization": (cfg.get("metadata") or {}).get("author_organization"),
        **hw,
        "agent_timeout_s": (cfg.get("agent") or {}).get("timeout_sec"),
        "verifier_timeout_s": verifier.get("timeout_sec"),
        "verifier_mode": verifier.get("environment_mode", "shared"),
        "artifacts": cfg.get("artifacts") or [],
        "bases": bases,
        "size_mb": round(size / 2**20, 1),
        "flags": flags,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Triage a Harbor dataset before porting it to karotte."
    )
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--json", type=Path, help="also write the rows as JSON here")
    args = ap.parse_args()
    task_dirs = sorted({p.parent for p in args.dataset.rglob("task.toml")})
    if not task_dirs:
        sys.exit(f"No task.toml under {args.dataset}")
    rows = [inspect(d) for d in task_dirs]
    print("| task | verdict | cpus/mem MB/disk MB/gpus | agent s | verifier | flags |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        hw = f"{r['cpus']}/{r['memory_mb']}/{r['storage_mb']}/{r['gpus']}"
        print(
            f"| {r['name']} | {r['verdict']} | {hw} | {r['agent_timeout_s']} | "
            + f"{r['verifier_mode']} | {', '.join(r['flags']) or '-'} |"
        )
    counts = {
        v: sum(r["verdict"] == v for r in rows)
        for v in ("portable", "needs-adaptation", "skip-or-defer")
    }
    print(f"\n{len(rows)} tasks: " + ", ".join(f"{n} {v}" for v, n in counts.items()))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
