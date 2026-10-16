"""Interrupt a running transcribe.py process."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from pathlib import Path

from config import PID_PATH
from errors import (
    EXIT_OK,
    NO_TRANSCRIPTION,
    AppError,
    could_not_read,
    could_not_read_pid,
    guard,
    info,
)


def command_for(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stop a running transcription.")
    parser.add_argument("--pid-file", type=Path, default=PID_PATH)
    args = parser.parse_args(argv)

    try:
        pid = int(args.pid_file.read_text(encoding="utf-8").strip())
    except FileNotFoundError as exc:
        raise AppError(NO_TRANSCRIPTION) from exc
    except UnicodeDecodeError as exc:
        raise AppError(could_not_read_pid(args.pid_file)) from exc
    except OSError as exc:
        raise AppError(could_not_read(args.pid_file)) from exc
    except ValueError as exc:
        raise AppError(could_not_read_pid(args.pid_file)) from exc

    command = command_for(pid)
    if "transcribe.py" not in command:
        args.pid_file.unlink(missing_ok=True)
        raise AppError(NO_TRANSCRIPTION)

    os.kill(pid, signal.SIGINT)
    info(f"Sent an interrupt to process {pid}.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(guard(main))
