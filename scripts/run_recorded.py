"""Run a command while persisting console output and a machine-readable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-root", default=str(PROJECT_ROOT / "reports" / "run_logs"))
    parser.add_argument("--cwd", default=str(PROJECT_ROOT))
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, dot, underscore, and hyphen")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    cwd = Path(args.cwd).resolve()
    log_root = Path(args.log_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{args.run_id}.log"
    manifest_path = log_root / f"{args.run_id}.json"

    started = _utc_now()
    started_perf = time.perf_counter()
    commit = _git_value("rev-parse", "HEAD")
    dirty = bool(_git_value("status", "--porcelain"))
    display_command = subprocess.list2cmdline(command)

    header = (
        f"RUN_ID: {args.run_id}\n"
        f"STARTED_UTC: {started.isoformat()}\n"
        f"CWD: {cwd}\n"
        f"GIT_COMMIT: {commit}\n"
        f"GIT_DIRTY: {dirty}\n"
        f"PROCESS_SEED: {args.seed}\n"
        f"COMMAND: {display_command}\n"
        + "=" * 80
        + "\n"
    )
    print(header, end="")

    with log_path.open("w", encoding="utf-8", newline="") as log:
        log.write(header)
        log.flush()
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env["PYTHONHASHSEED"] = str(args.seed)
        child_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            exit_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exit_code = 130
            interrupted = "RUN_INTERRUPTED: true\n"
            print(interrupted, end="")
            log.write(interrupted)

        ended = _utc_now()
        duration = time.perf_counter() - started_perf
        footer = (
            "=" * 80
            + "\n"
            + f"ENDED_UTC: {ended.isoformat()}\n"
            + f"DURATION_SECONDS: {duration:.6f}\n"
            + f"EXIT_CODE: {exit_code}\n"
        )
        print(footer, end="")
        log.write(footer)

    manifest = {
        "run_id": args.run_id,
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "duration_seconds": round(duration, 6),
        "exit_code": exit_code,
        "cwd": str(cwd),
        "command": command,
        "command_display": display_command,
        "git_commit": commit,
        "git_dirty_at_start": dirty,
        "process_seed": args.seed,
        "pythonhashseed": child_env["PYTHONHASHSEED"],
        "cublas_workspace_config": child_env["CUBLAS_WORKSPACE_CONFIG"],
        "runner_python": sys.executable,
        "log_path": str(log_path),
        "log_sha256": _sha256(log_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
