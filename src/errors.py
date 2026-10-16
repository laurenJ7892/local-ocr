"""Exit codes and terminal messages shared by the scripts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_STOPPED = 130

NO_TRANSCRIPTION = "No transcription is running."
STOPPING_IMMEDIATELY = "\nStopping immediately."
STOP_REQUESTED = (
    "\nStop requested. The current page will not be saved. "
    "Press Ctrl+C again to stop immediately."
)
STOPPED = "Stopped. Finished pages are saved. Run the same command to continue."
FINISHED = "Finished."


class AppError(Exception):
    """A problem the user can fix. The scripts print it without a traceback."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


# What: Prints a progress line on standard output.
# Needs: The message.
# Returns: Nothing.
def info(message: str) -> None:
    print(message, file=sys.stdout)


# What: Prints a problem line on standard error.
# Needs: The message.
# Returns: Nothing.
def error(message: str) -> None:
    print(message, file=sys.stderr)


# What: Runs a command and turns an AppError into an exit code.
# Needs: A main function that accepts optional argv and returns an exit code.
# Returns: The exit code. A broken pipe counts as success.
def guard(main: Callable[[list[str] | None], int], argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except AppError as exc:
        error(str(exc))
        return exc.code
    except BrokenPipeError:
        return EXIT_OK


# What: Describes a file that could not be opened.
# Needs: The file path.
# Returns: The error sentence.
def could_not_read(path: Path) -> str:
    return f"Could not read {path}."


# What: Describes a pid file whose contents are not a process id.
# Needs: The pid file path.
# Returns: The error sentence.
def could_not_read_pid(path: Path) -> str:
    return f"Could not read a process id from {path}."


# What: Describes a file that is not UTF-8 text.
# Needs: The file path.
# Returns: The error sentence.
def not_utf8(path: Path) -> str:
    return f"Could not read {path} as UTF-8 text."


# What: Describes a file that could not be written.
# Needs: The file path.
# Returns: The error sentence.
def could_not_write(path: Path) -> str:
    return f"Could not write {path}."


# What: Reads a text file, or raises AppError when that fails.
# Needs: A file path.
# Returns: The file contents.
def read_output(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise AppError(not_utf8(path)) from exc
    except OSError as exc:
        raise AppError(could_not_read(path)) from exc


# What: Writes a text file, creating parent folders when needed.
# Needs: The file path and the contents.
# Returns: Nothing. Raises AppError when the write fails.
def write_output(path: Path, contents: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    except OSError as exc:
        raise AppError(could_not_write(path)) from exc
