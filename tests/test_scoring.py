import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

import evaluate
import scoring
import text
import transcribe


class TextTests(unittest.TestCase):
    def test_case_and_quotes_fold_only_in_the_normalized_view(self):
        case = scoring.page_scores("Hello", "hello")
        self.assertGreater(case["strict_cer"], 0)
        self.assertEqual(case["normalized_cer"], 0)

        quotes = scoring.page_scores("it’s", "it's")
        self.assertGreater(quotes["strict_cer"], 0)
        self.assertEqual(quotes["normalized_cer"], 0)

    def test_extra_spaces_fold_and_line_breaks_stay_in_character_error(self):
        spaces = scoring.page_scores("Hello  world", "Hello world")
        self.assertEqual(spaces["normalized_cer"], 0)
        self.assertGreater(spaces["strict_cer"], 0)

        lines = scoring.page_scores("Hello\nworld", "Hello world")
        self.assertGreater(lines["normalized_cer"], 0)
        self.assertEqual(lines["normalized_wer"], 0)

    def test_scoring_views_are_shared_by_page_and_corpus(self):
        views = text.scoring_views("Hello  it’s\n")
        pair = scoring.page_scores("Hello  it’s\n", "hello it's")
        corpus = scoring.corpus_scores([("Hello  it’s\n", "hello it's")])
        self.assertIn("strict_cer", views)
        self.assertEqual(pair["normalized_cer"], corpus["normalized_cer"])


class DecisionTests(unittest.TestCase):
    def test_bands(self):
        perfect = evaluate.recommendation(
            "glm-ocr",
            {"glm-ocr": {"normalized_cer": 0.0, "normalized_wer": 0.0, "pages": 2}},
            {"non_alphanumeric_edit_share": 0.0, "concentrated_letter_shapes": False},
        )
        self.assertEqual(perfect["band"], "good_enough")
        self.assertIn("match the transcripts", perfect["text"])

        punctuation = evaluate.recommendation(
            "glm-ocr",
            {"glm-ocr": {"normalized_cer": 0.04, "normalized_wer": 0.05, "pages": 2}},
            {"non_alphanumeric_edit_share": 0.8, "concentrated_letter_shapes": False},
        )
        self.assertIn("punctuation or line breaks", punctuation["text"])

        mid = evaluate.recommendation(
            "qwen3.5-9b",
            {
                "qwen3.5-9b": {"normalized_cer": 0.12, "normalized_wer": 0.2, "pages": 3},
                "glm-ocr": {"normalized_cer": 0.13, "normalized_wer": 0.22, "pages": 3},
            },
            {"non_alphanumeric_edit_share": 0.1, "concentrated_letter_shapes": True},
        )
        self.assertEqual(mid["band"], "finetune_plausible")
        self.assertIn("cheaper", mid["text"])

        high = evaluate.recommendation(
            "glm-ocr",
            {"glm-ocr": {"normalized_cer": 0.31, "normalized_wer": 0.45, "pages": 3}},
            {"non_alphanumeric_edit_share": 0.1, "concentrated_letter_shapes": False},
        )
        self.assertEqual(high["band"], "finetune_justified")

    def test_a_run_of_letter_errors_counts_as_letter_shapes(self):
        profile = scoring.error_profile([("abcd " * 2, "wxyz " * 2)])
        pairs = {(item["from"], item["to"]) for item in profile["letter_substitutions"]}
        self.assertIn(("a", "w"), pairs)
        self.assertIn(("d", "z"), pairs)
        self.assertTrue(profile["concentrated_letter_shapes"])

    def test_disjoint_pages_are_not_ranked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages = root / "pages"
            truth = root / "truth"
            preds = root / "preds"
            pages.mkdir()
            truth.mkdir()
            (pages / "a.jpg").write_bytes(b"x")
            (pages / "b.jpg").write_bytes(b"y")
            (truth / "a.txt").write_text("alpha\n", encoding="utf-8")
            (truth / "b.txt").write_text("beta\n", encoding="utf-8")
            (preds / "glm-ocr").mkdir(parents=True)
            (preds / "qwen3.5-9b").mkdir(parents=True)
            (preds / "glm-ocr" / "a.txt").write_text("alpha\n", encoding="utf-8")
            (preds / "qwen3.5-9b" / "b.txt").write_text("beta\n", encoding="utf-8")
            result = evaluate.evaluate(truth, preds, pages)
        self.assertIsNone(result["winner"])
        self.assertEqual(result["recommendation"]["band"], "disjoint_pages")
        self.assertTrue(result["disjoint"])
        report = evaluate.build_report(result)
        self.assertIn("not ranked", report)


class TranscribeHelperTests(unittest.TestCase):
    def test_newer_photo_is_transcribed_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            photo = folder / "page.jpg"
            prediction = folder / "page.txt"
            photo.write_bytes(b"old")
            prediction.write_text("kept\n", encoding="utf-8")
            os.utime(prediction, (1_000, 1_000))
            os.utime(photo, (2_000, 2_000))
            self.assertFalse(transcribe.already_done(prediction, photo))
            os.utime(photo, (500, 500))
            self.assertTrue(transcribe.already_done(prediction, photo))

    def test_blank_prediction_is_not_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            photo = folder / "page.jpg"
            prediction = folder / "page.txt"
            photo.write_bytes(b"photo")
            prediction.write_text("\n", encoding="utf-8")
            self.assertFalse(transcribe.already_done(prediction, photo))

    def test_display_path_keeps_paths_outside_the_repo(self):
        outside = Path("/tmp/handwriting-predictions/page.txt")
        self.assertEqual(transcribe.display_path(outside), str(outside))

    def test_resize_limits_the_long_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "big.png"
            dest = Path(tmp) / "small.jpg"
            Image.new("RGB", (3200, 800), "white").save(source)
            transcribe.resize_long_edge(source, dest, 1600)
            with Image.open(dest) as image:
                self.assertEqual(image.size, (1600, 400))


if __name__ == "__main__":
    unittest.main()
