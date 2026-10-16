"""Turn scores into a fine-tune decision and a markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    CER_FINETUNE_PLAUSIBLE,
    CER_GOOD_ENOUGH,
    CLOSE_CER,
    EXCERPT_LIMIT,
    MODEL_SLUGS,
    PAGES_DIR,
    PREDICTIONS_DIR,
    REPORT_PATH,
    SCORES_PATH,
    TRUTH_DIR,
)
from errors import EXIT_OK, guard, info, write_output
from scoring import Aggregates, compare
from text import strict_text


# What: Formats the two headline rates for the decision sentence.
# Needs: The winning slug, its normalized CER, and its normalized WER.
# Returns: A sentence ending with a period.
def rate_summary(winner: str, cer: float, wer: float) -> str:
    return (
        f"{winner} has a normalized character error rate of {cer:.1%} "
        f"and a word error rate of {wer:.1%}."
    )


# What: Adds the cheaper-model note when GLM-OCR won or is within 2 points.
# Needs: The winner, all model aggregates, and the winner's normalized CER.
# Returns: A leading-space sentence, or an empty string when GLM-OCR is not close.
def glm_adaptation_note(winner: str, aggregates: Aggregates, cer: float) -> str:
    glm_cer = aggregates.get("glm-ocr", {}).get("normalized_cer")
    close_to_glm = winner == "glm-ocr" or (
        isinstance(glm_cer, float) and abs(glm_cer - cer) <= CLOSE_CER
    )
    if not close_to_glm:
        return ""
    return (
        " GLM-OCR is the cheaper model to adapt if it is close, "
        "because it is under 1B parameters."
    )


# What: Joins the shared rate sentence to the part that differs by band.
# Needs: The winner, both rates, and the band ending. The ending starts with a space.
# Returns: The full decision paragraph.
def decision_text(winner: str, cer: float, wer: float, ending: str) -> str:
    return f"{rate_summary(winner, cer, wer)}{ending}"


# What: Writes the ending for a model at or below 8% character error.
# Needs: The normalized CER and the error profile.
# Returns: The ending, starting with a space.
def good_enough_ending(cer: float, profile: dict[str, object]) -> str:
    share = profile.get("non_alphanumeric_edit_share")
    mostly_punctuation = isinstance(share, float) and share >= 0.5
    if cer == 0:
        detail = " The normalized transcriptions match the transcripts."
    elif mostly_punctuation:
        detail = " The misses are mostly punctuation or line breaks."
    else:
        detail = " Some misses change letters, and the overall rate is still in this band."
    return f"{detail} That base model is good enough."


# What: Writes the ending for a model between 8% and 20% character error.
# Needs: The error profile and the GLM note.
# Returns: The ending, starting with a space.
def finetune_plausible_ending(profile: dict[str, object], glm_note: str) -> str:
    if profile.get("concentrated_letter_shapes"):
        shape = " The errors cluster on a few letter shapes."
    else:
        shape = " The errors are spread across ordinary words."
    return (
        f"{shape} Fine-tuning is plausible.{glm_note} "
        "A later project would need on the order of a hundred labeled pages."
    )


# What: Writes the ending for a model above 20% character error.
# Needs: The winning slug and the GLM note.
# Returns: The ending, starting with a space.
def finetune_justified_ending(winner: str, glm_note: str) -> str:
    return (
        " Errors on clear photos are high enough that fine-tuning on your handwriting "
        f"is justified, starting from {winner}.{glm_note}"
    )


# What: Turns the winning scores into a fine-tune recommendation.
# Needs: The winning slug or None, per-model aggregates, and that model's error profile.
# Returns: A band name and the sentence to print in the report.
def recommendation(
    winner: str | None,
    aggregates: Aggregates,
    profile: dict[str, object],
) -> dict[str, str]:
    if winner is None:
        return {
            "band": "insufficient_data",
            "text": (
                "There are no pages with both a typed transcript and a prediction, "
                "so there is no fine-tune decision yet."
            ),
        }
    cer = float(aggregates[winner]["normalized_cer"])
    wer = float(aggregates[winner]["normalized_wer"])
    glm_note = glm_adaptation_note(winner, aggregates, cer)
    if cer <= CER_GOOD_ENOUGH:
        return {
            "band": "good_enough",
            "text": decision_text(winner, cer, wer, good_enough_ending(cer, profile)),
        }
    if cer <= CER_FINETUNE_PLAUSIBLE:
        return {
            "band": "finetune_plausible",
            "text": decision_text(
                winner, cer, wer, finetune_plausible_ending(profile, glm_note)
            ),
        }
    return {
        "band": "finetune_justified",
        "text": decision_text(winner, cer, wer, finetune_justified_ending(winner, glm_note)),
    }



# What: Writes the fine-tune sentence for a scored comparison.
# Needs: The dictionary from compare, including the winner and whether the pages are disjoint.
# Returns: A band name and the sentence to print in the report.
def decision_for(result: dict) -> dict[str, str]:
    if result["disjoint"]:
        return {
            "band": "disjoint_pages",
            "text": (
                "The two models finished different pages, so there is no "
                "head-to-head fine-tune decision yet."
            ),
        }
    return recommendation(result["winner"], result["aggregates"], result["error_profile"])



# What: Scores both models and adds the fine-tune decision.
# Needs: Directories for transcripts, predictions, and page photos.
# Returns: The comparison plus a recommendation band and sentence.
def evaluate(truth_dir: Path, predictions_dir: Path, pages_dir: Path) -> dict[str, object]:
    result = compare(truth_dir, predictions_dir, pages_dir)
    result["recommendation"] = decision_for(result)
    return result



# What: Shortens a long transcript for the report.
# Needs: Raw page text.
# Returns: The strict text, cut at the excerpt limit when it is longer than that.
def excerpt(text: str) -> str:
    cleaned = strict_text(text)
    if len(cleaned) <= EXCERPT_LIMIT:
        return cleaned
    return cleaned[:EXCERPT_LIMIT].rstrip() + "\n…"


# What: Formats an error rate for the markdown table.
# Needs: A rate from 0 to 1, or None.
# Returns: A percent such as "12.0%", or an em dash when the rate is missing.
def percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}"


# What: Builds one markdown row for a model.
# Needs: The model slug and its aggregate scores, which may be missing.
# Returns: A markdown table row.
def model_row(slug: str, aggregate: dict | None) -> str:
    if not aggregate or not aggregate["pages"]:
        return f"| {slug} | 0 | — | — | — | — |"
    return "| {slug} | {pages} | {ncer} | {nwer} | {scer} | {swer} |".format(
        slug=slug,
        pages=aggregate["pages"],
        ncer=percent(aggregate["normalized_cer"]),
        nwer=percent(aggregate["normalized_wer"]),
        scer=percent(aggregate["strict_cer"]),
        swer=percent(aggregate["strict_wer"]),
    )


# What: Builds the comparison table and the sentence under it.
# Needs: The full evaluation result.
# Returns: Markdown lines for the head-to-head section.
def head_to_head_lines(result: dict[str, object]) -> list[str]:
    aggregates: dict = result["aggregates"]
    lines = [
        "## Head to head",
        "",
        "| Model | Pages | Normalized CER | Normalized WER | Strict CER | Strict WER |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for slug in MODEL_SLUGS:
        lines.append(model_row(slug, aggregates.get(slug)))
    lines.append("")
    if result.get("disjoint"):
        lines.append(
            "The models finished different pages, so these rates are not ranked against each other."
        )
    elif result["winner"]:
        lines.append(f"Lower normalized character error rate: **{result['winner']}**.")
    else:
        lines.append("No model has a scored page yet.")
    compared = result["compared_pages"]
    if compared is not None:
        lines.append("")
        noun = "page" if compared == 1 else "pages"
        lines.append(f"The rates above use the {compared} {noun} both models finished.")
    return lines


# What: Builds the side-by-side excerpts for the pages worth reading.
# Needs: The page records and the stems to show.
# Returns: Markdown lines, including each transcript and prediction.
def side_by_side_lines(pages: dict, shown: list[str]) -> list[str]:
    lines = ["## Side by side", ""]
    if not shown:
        lines.append("No scored pages to show.")
    for stem in shown:
        page = pages[stem]
        lines.extend(
            [
                f"### {stem}",
                "",
                "Transcript:",
                "",
                "```text",
                excerpt(page["reference"]),
                "```",
                "",
            ]
        )
        for slug in MODEL_SLUGS:
            prediction = page["predictions"].get(slug)
            if prediction is None:
                continue
            score = percent(page["scores"][slug]["normalized_cer"])
            lines.extend(
                [
                    f"{slug} (normalized CER {score}):",
                    "",
                    "```text",
                    excerpt(prediction),
                    "```",
                    "",
                ]
            )
    return lines


# What: Turns an evaluation result into the markdown report.
# Needs: The dictionary returned by evaluate.
# Returns: The full report, ending with a newline.
def build_report(result: dict[str, object]) -> str:
    decision = result["recommendation"]
    disagreements = result["disagreements"]
    lines = ["# Handwriting transcription report", "", *head_to_head_lines(result), ""]
    lines.extend(["## Fine-tune decision", "", decision["text"], ""])
    lines.extend(["## Pages where the models disagree", ""])
    if disagreements:
        lines.append(", ".join(disagreements))
    else:
        lines.append("The normalized transcriptions agree on every shared page.")
    lines.extend(["", *side_by_side_lines(result["pages"], result["side_by_side"])])
    skipped = result["skipped"]
    if skipped:
        lines.extend(
            [
                "## Skipped",
                "",
                "These photos have no matching transcript, so they were not scored: "
                + ", ".join(skipped)
                + ".",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


# What: Keeps the fields the JSON scores file needs.
# Needs: The dictionary returned by evaluate.
# Returns: A JSON-ready dictionary without the full page text.
def scores_payload(result: dict[str, object]) -> dict[str, object]:
    pages_out = {}
    for stem, page in result["pages"].items():
        pages_out[stem] = page["scores"]
    return {
        "winner": result["winner"],
        "recommendation": result["recommendation"],
        "error_profile": result["error_profile"],
        "aggregates": result["aggregates"],
        "pages": pages_out,
        "disagreements": result["disagreements"],
    }


# What: Reads the command-line paths for a scoring run.
# Needs: Optional argv. Defaults to the project's data and output folders.
# Returns: The parsed arguments.
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score notebook transcriptions.")
    parser.add_argument("--pages", type=Path, default=PAGES_DIR)
    parser.add_argument("--ground-truth", type=Path, default=TRUTH_DIR)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--scores", type=Path, default=SCORES_PATH)
    return parser.parse_args(argv)


# What: Scores the pages on disk and writes the report.
# Needs: Optional argv. Uses the default folders when omitted.
# Returns: 0 after the report and scores file are written.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = evaluate(args.ground_truth, args.predictions, args.pages)
    report = build_report(result)
    write_output(args.report, report)
    write_output(args.scores, json.dumps(scores_payload(result), indent=2) + "\n")
    info(result["recommendation"]["text"])
    info(f"Wrote {args.report}")
    info(f"Wrote {args.scores}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(guard(main))
