"""Unit tests for transcription text views.

A class is the Jest `describe` block. Each `test_` method is an `it`.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import text


class IsEmpty(unittest.TestCase):
    def test_accepts_an_empty_string(self):
        self.assertTrue(text.is_empty(""))

    def test_rejects_spaces_and_words(self):
        self.assertFalse(text.is_empty(" "))
        self.assertFalse(text.is_empty("\n"))
        self.assertFalse(text.is_empty("hello"))


class StrictText(unittest.TestCase):
    def test_unifies_windows_and_old_mac_line_endings(self):
        self.assertEqual(text.strict_text("Hello\r\nworld"), "Hello\nworld")
        self.assertEqual(text.strict_text("Hello\rworld"), "Hello\nworld")

    def test_strips_only_newlines_at_the_edges(self):
        self.assertEqual(text.strict_text("\n\nHello \n"), "Hello ")
        self.assertEqual(text.strict_text("Hello\n\nworld"), "Hello\n\nworld")

    def test_keeps_case_spelling_and_curly_quotes(self):
        self.assertEqual(text.strict_text("It’s Teh"), "It’s Teh")


class NormalizedText(unittest.TestCase):
    def test_folds_case_quotes_and_dashes(self):
        self.assertEqual(text.normalized_text("It’s Teh—end"), "it's teh-end")

    def test_collapses_horizontal_space_and_keeps_line_breaks(self):
        self.assertEqual(text.normalized_text("Hello \t  world"), "hello world")
        self.assertEqual(text.normalized_text("Hello \n world"), "hello\nworld")

    def test_collapses_long_blank_lines_and_outer_space(self):
        self.assertEqual(text.normalized_text("  Hello\n\n\n\nworld  "), "hello\n\nworld")

    def test_keeps_spelling(self):
        self.assertEqual(text.normalized_text("teh"), "teh")


class ForWords(unittest.TestCase):
    def test_turns_line_breaks_into_word_spaces(self):
        self.assertEqual(text.for_words("Hello\n\nworld"), "Hello world")

    def test_collapses_mixed_whitespace(self):
        self.assertEqual(text.for_words(" \tHello  \n world "), "Hello world")


class ScoringViews(unittest.TestCase):
    def test_builds_each_error_rate_string(self):
        views = text.scoring_views("It’s  Teh\r\nnext")

        self.assertEqual(
            views,
            {
                "strict_cer": "It’s  Teh\nnext",
                "strict_wer": "It’s Teh next",
                "normalized_cer": "it's teh\nnext",
                "normalized_wer": "it's teh next",
            },
        )


if __name__ == "__main__":
    unittest.main()
