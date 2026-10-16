"""Unit tests for shared exit codes and terminal messages.

A class is the Jest `describe` block. Each `test_` method is an `it`.
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import errors
import scoring


class Guard(unittest.TestCase):
    def test_prints_an_app_error_and_returns_its_code(self):
        def main(argv=None):
            raise errors.AppError(errors.NO_TRANSCRIPTION)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = errors.guard(main)
        self.assertEqual(code, errors.EXIT_ERROR)
        self.assertEqual(stderr.getvalue(), "No transcription is running.\n")

    def test_returns_the_code_from_main(self):
        def main(argv=None):
            errors.info(errors.FINISHED)
            return errors.EXIT_OK

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = errors.guard(main)
        self.assertEqual(code, errors.EXIT_OK)
        self.assertEqual(stdout.getvalue(), "Finished.\n")

    def test_a_broken_pipe_counts_as_success(self):
        def main(argv=None):
            raise BrokenPipeError()

        self.assertEqual(errors.guard(main), errors.EXIT_OK)


class ReadAndWrite(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "page.txt"
            errors.write_output(path, "hello\n")
            self.assertEqual(errors.read_output(path), "hello\n")

    def test_missing_file_is_an_app_error(self):
        missing = Path("/tmp/handwriting-missing-page.txt")
        with self.assertRaises(errors.AppError) as caught:
            scoring.read_text(missing)
        self.assertEqual(str(caught.exception), f"Could not read {missing}.")
        self.assertEqual(caught.exception.code, errors.EXIT_ERROR)

    def test_non_utf8_file_is_an_app_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.txt"
            path.write_bytes(b"\xff")
            with self.assertRaises(errors.AppError) as caught:
                scoring.read_text(path)
        self.assertEqual(str(caught.exception), f"Could not read {path} as UTF-8 text.")


if __name__ == "__main__":
    unittest.main()
