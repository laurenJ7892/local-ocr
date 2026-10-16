"""Measure transcriptions against typed notebook transcripts."""

from __future__ import annotations

import difflib
from collections import defaultdict
from pathlib import Path

import jiwer

from config import MODEL_SLUGS, SIDE_BY_SIDE_LIMIT
from errors import read_output
from text import is_empty, normalized_text, scoring_views

METRIC_KIND = {
    "strict_cer": "cer",
    "strict_wer": "wer",
    "normalized_cer": "cer",
    "normalized_wer": "wer",
}
Pair = tuple[str, str]
Aggregates = dict[str, dict[str, float | int | None]]


# What: Reads one text file as a string.
# Needs: A file path.
# Returns: The file contents. Raises AppError when the file cannot be read.
def read_text(path: Path) -> str:
    return read_output(path)


# What: Lists file names, without extensions, in one folder.
# Needs: A directory and the extensions to keep, such as {".txt"}.
# Returns: Sorted stems. An empty list when the directory is missing.
def list_stems(directory: Path, suffixes: set[str]) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


# What: Measures one reference against one prediction.
# Needs: Both strings, already prepared for scoring, and "cer" or "wer".
# Returns: 0 when both are empty, 1 when only the reference is empty, otherwise the rate.
def safe_rate(reference: str, hypothesis: str, kind: str) -> float:
    if is_empty(reference) and is_empty(hypothesis):
        return 0.0
    if is_empty(reference):
        return 1.0
    measure = jiwer.cer if kind == "cer" else jiwer.wer
    return float(measure(reference, hypothesis))


# What: Drops pairs that cannot be scored, then runs one corpus measure.
# Needs: Parallel lists of prepared references and predictions, and "cer" or "wer".
# Returns: The corpus error rate.
def corpus_rate(references: list[str], hypotheses: list[str], kind: str) -> float:
    usable = [
        (reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses)
        if not is_empty(reference) or not is_empty(hypothesis)
    ]
    if not usable:
        return 0.0
    if all(is_empty(reference) for reference, _ in usable):
        return 1.0
    kept = [
        (reference, hypothesis)
        for reference, hypothesis in usable
        if not is_empty(reference)
    ]
    measure = jiwer.process_characters if kind == "cer" else jiwer.process_words
    output = measure(
        [reference for reference, _ in kept],
        [hypothesis for _, hypothesis in kept],
    )
    return float(output.cer if kind == "cer" else output.wer)


# What: Scores every strict and normalized view with the same rate function.
# Needs: Raw references, raw predictions, and a function that scores one view.
# Returns: strict_cer, strict_wer, normalized_cer, and normalized_wer.
def rates_for(references: list[str], hypotheses: list[str], rate) -> dict[str, float]:
    ref_views = [scoring_views(text) for text in references]
    hyp_views = [scoring_views(text) for text in hypotheses]
    return {
        name: rate(
            [view[name] for view in ref_views],
            [view[name] for view in hyp_views],
            kind,
        )
        for name, kind in METRIC_KIND.items()
    }


# What: Scores a single page from its first prepared string.
# Needs: One-item lists of a prepared reference and prediction, and "cer" or "wer".
# Returns: The page error rate.
def page_rate(references: list[str], hypotheses: list[str], kind: str) -> float:
    return safe_rate(references[0], hypotheses[0], kind)


# What: Scores one transcript against one prediction.
# Needs: The typed transcript and the model text.
# Returns: The four page error rates.
def page_scores(reference: str, hypothesis: str) -> dict[str, float]:
    return rates_for([reference], [hypothesis], page_rate)


# What: Scores a whole set of pages as one corpus.
# Needs: Pairs of (transcript, prediction).
# Returns: The four corpus rates plus a page count. Rates are None when there are no pairs.
def corpus_scores(pairs: list[Pair]) -> dict[str, float | int | None]:
    if not pairs:
        return {name: None for name in METRIC_KIND} | {"pages": 0}
    references = [reference for reference, _ in pairs]
    hypotheses = [hypothesis for _, hypothesis in pairs]
    return rates_for(references, hypotheses, corpus_rate) | {"pages": len(pairs)}


# What: Finds equal-length pieces where the prediction replaced the transcript.
# Needs: The original span and the predicted span from one jiwer substitution.
# Returns: Pairs of equal-length source and target strings.
def replacement_spans(source: str, target: str) -> list[tuple[str, str]]:
    if len(source) == len(target):
        return [(source, target)]
    spans = []
    matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            spans.append((source[i1:i2], target[j1:j2]))
    return spans


# What: Breaks a substitution span into single-letter confusions.
# Needs: The transcript span and the prediction span.
# Returns: Pairs of letters that differ. A run such as "abcd" to "wxyz" stays four pairs.
def letter_substitutions(source: str, target: str) -> list[tuple[str, str]]:
    pairs = []
    for span_source, span_target in replacement_spans(source, target):
        for source_char, target_char in zip(span_source, span_target):
            if (
                source_char != target_char
                and source_char.isalpha()
                and target_char.isalpha()
            ):
                pairs.append((source_char, target_char))
    return pairs


# What: Keeps normalized pairs that have a transcript to score against.
# Needs: Raw (transcript, prediction) pairs.
# Returns: Parallel lists of normalized transcript and prediction.
def pairs_with_transcript(pairs: list[Pair]) -> tuple[list[str], list[str]]:
    references = []
    hypotheses = []
    for reference, hypothesis in pairs:
        reference = normalized_text(reference)
        hypothesis = normalized_text(hypothesis)
        if is_empty(reference):
            continue
        references.append(reference)
        hypotheses.append(hypothesis)
    return references, hypotheses


# What: The profile used when there is nothing to compare.
# Needs: Nothing.
# Returns: An empty error profile.
def empty_error_profile() -> dict[str, object]:
    return {
        "non_alphanumeric_edit_share": None,
        "letter_substitutions": [],
        "concentrated_letter_shapes": False,
    }


# What: Measures how many edits are punctuation, spaces, or line breaks.
# Needs: jiwer substitution, insertion, and deletion counts.
# Returns: That share, from 0 to 1. Returns 0 when there are no edits.
def non_word_edit_share(substitutions, insertions, deletions) -> float:
    edits = 0
    non_word_edits = 0
    edit_counts = (
        list(substitutions.items()) + list(insertions.items()) + list(deletions.items())
    )
    for key, count in edit_counts:
        changed = "".join(key) if isinstance(key, tuple) else key
        edits += count
        if not is_empty(changed) and all(not character.isalnum() for character in changed):
            non_word_edits += count
    if edits == 0:
        return 0.0
    return non_word_edits / edits


# What: Counts letter confusions and lists the most common ones first.
# Needs: jiwer substitution counts, including multi-character spans.
# Returns: (source letter, target letter, count) tuples, most frequent first.
def ranked_letter_pairs(substitutions) -> list[tuple[str, str, int]]:
    letter_counts: dict[tuple[str, str], int] = defaultdict(int)
    for (source, target), count in substitutions.items():
        for source_char, target_char in letter_substitutions(source, target):
            letter_counts[(source_char, target_char)] += count
    ranked = [
        (source, target, count)
        for (source, target), count in letter_counts.items()
    ]
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


# What: Decides whether a few letter pairs explain most letter errors.
# Needs: Ranked (source, target, count) letter pairs.
# Returns: True when at least four letter errors exist and the top three cover half of them.
def letters_are_concentrated(letter_pairs: list[tuple[str, str, int]]) -> bool:
    letter_total = sum(count for _, _, count in letter_pairs)
    if letter_total < 4:
        return False
    top = sum(count for _, _, count in letter_pairs[:3])
    return top / letter_total >= 0.5


# What: Describes the kinds of mistakes in a set of pages.
# Needs: Pairs of (transcript, prediction).
# Returns: The punctuation share, the top letter substitutions, and whether they cluster.
def error_profile(pairs: list[Pair]) -> dict[str, object]:
    references, hypotheses = pairs_with_transcript(pairs)
    if not references:
        return empty_error_profile()
    output = jiwer.process_characters(references, hypotheses)
    substitutions, insertions, deletions = jiwer.collect_error_counts(output)
    letter_pairs = ranked_letter_pairs(substitutions)
    return {
        "non_alphanumeric_edit_share": non_word_edit_share(
            substitutions, insertions, deletions
        ),
        "letter_substitutions": [
            {"from": source, "to": target, "count": count}
            for source, target, count in letter_pairs[:8]
        ],
        "concentrated_letter_shapes": letters_are_concentrated(letter_pairs),
    }


# What: Picks the model with the lower normalized character error rate.
# Needs: Per-model aggregate scores. A tie goes to glm-ocr.
# Returns: The winning model slug, or None when no model has scored pages.
def choose_winner(aggregates: Aggregates) -> str | None:
    ranked = [
        (slug, aggregate["normalized_cer"])
        for slug, aggregate in aggregates.items()
        if aggregate.get("pages") and aggregate.get("normalized_cer") is not None
    ]
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[1], 0 if item[0] == "glm-ocr" else 1))
    return ranked[0][0]


# What: Loads each model's saved prediction files.
# Needs: The predictions directory, with one subdirectory per model slug.
# Returns: A mapping of slug to stem to prediction text.
def load_predictions(predictions_dir: Path) -> dict[str, dict[str, str]]:
    predictions: dict[str, dict[str, str]] = {}
    for slug in MODEL_SLUGS:
        model_dir = predictions_dir / slug
        predictions[slug] = {}
        if not model_dir.is_dir():
            continue
        for path in sorted(model_dir.glob("*.txt")):
            predictions[slug][path.stem] = read_text(path)
    return predictions


# What: Scores every transcript that at least one model predicted.
# Needs: The ground-truth directory and the loaded predictions.
# Returns: Truth stems, scored stems, per-page records, and per-model text pairs.
def score_pages(
    truth_dir: Path,
    predictions: dict[str, dict[str, str]],
) -> tuple[list[str], list[str], dict[str, dict[str, object]], dict[str, list[Pair]]]:
    truth_stems = list_stems(truth_dir, {".txt"})
    scored_stems = [
        stem
        for stem in truth_stems
        if any(stem in predictions[slug] for slug in MODEL_SLUGS)
    ]
    pages: dict[str, dict[str, object]] = {}
    per_model_pairs: dict[str, list[Pair]] = {slug: [] for slug in MODEL_SLUGS}
    for stem in scored_stems:
        reference = read_text(truth_dir / f"{stem}.txt")
        page_predictions = {}
        page_scores_by_model = {}
        for slug in MODEL_SLUGS:
            if stem not in predictions[slug]:
                continue
            hypothesis = predictions[slug][stem]
            page_predictions[slug] = hypothesis
            page_scores_by_model[slug] = page_scores(reference, hypothesis)
            per_model_pairs[slug].append((reference, hypothesis))
        pages[stem] = {
            "reference": reference,
            "predictions": page_predictions,
            "scores": page_scores_by_model,
        }
    return truth_stems, scored_stems, pages, per_model_pairs


# What: Finds pages both models predicted.
# Needs: Scored stems and the loaded predictions.
# Returns: Stems present for every model, in scored order.
def shared_stems(scored_stems: list[str], predictions: dict[str, dict[str, str]]) -> list[str]:
    return [
        stem
        for stem in scored_stems
        if all(stem in predictions[slug] for slug in MODEL_SLUGS)
    ]


# What: Uses the shared pages when both models finished some of the same work.
# Needs: Page records, per-model pairs, predictions, and scored stems.
# Returns: Aggregates, the pairs used for the decision, decision stems, whether the
# sets are disjoint, and the shared stems.
def align_models(
    pages: dict[str, dict[str, object]],
    per_model_pairs: dict[str, list[Pair]],
    predictions: dict[str, dict[str, str]],
    scored_stems: list[str],
) -> tuple[Aggregates, dict[str, list[Pair]], list[str], bool, list[str]]:
    aggregates = {slug: corpus_scores(per_model_pairs[slug]) for slug in MODEL_SLUGS}
    shared = shared_stems(scored_stems, predictions)
    both_have_pages = all(aggregates[slug]["pages"] for slug in MODEL_SLUGS)
    disjoint = both_have_pages and not shared
    if not (shared and both_have_pages):
        return aggregates, per_model_pairs, scored_stems, disjoint, shared
    shared_pairs = {
        slug: [
            (pages[stem]["reference"], pages[stem]["predictions"][slug])
            for stem in shared
        ]
        for slug in MODEL_SLUGS
    }
    for slug in MODEL_SLUGS:
        aggregates[slug] = corpus_scores(shared_pairs[slug])
    return aggregates, shared_pairs, shared, disjoint, shared


# What: Lists pages whose normalized predictions differ.
# Needs: Stems in the decision set and the per-page records.
# Returns: Those stems, in decision order.
def disagreeing_stems(decision_stems: list[str], pages: dict[str, dict[str, object]]) -> list[str]:
    disagreements = []
    for stem in decision_stems:
        present = [slug for slug in MODEL_SLUGS if slug in pages[stem]["predictions"]]
        if len(present) < 2:
            continue
        texts = {normalized_text(pages[stem]["predictions"][slug]) for slug in present}
        if len(texts) > 1:
            disagreements.append(stem)
    return disagreements


# What: Chooses which pages to print in the side-by-side section.
# Needs: Decision stems, page records, the winner, and the disagreeing stems.
# Returns: Up to eight stems, worst pages first, including disagreements.
def pages_to_show(
    decision_stems: list[str],
    pages: dict[str, dict[str, object]],
    winner: str | None,
    disagreements: list[str],
) -> list[str]:
    ranked = sorted(
        decision_stems,
        key=lambda stem: (
            pages[stem]["scores"].get(winner, {}).get("normalized_cer", 0.0)
            if winner
            else 0.0
        ),
        reverse=True,
    )
    shown: list[str] = []
    for stem in ranked:
        if stem in disagreements or len(shown) < 5:
            shown.append(stem)
        if len(shown) >= SIDE_BY_SIDE_LIMIT:
            break
    return shown


# What: Picks the winner and error profile without writing the decision sentence.
# Needs: Aggregates, the pairs each model is judged on, and whether the page sets are disjoint.
# Returns: The winner or None, and the error profile.
def judge(
    aggregates: Aggregates,
    decision_pairs: dict[str, list[Pair]],
    disjoint: bool,
) -> tuple[str | None, dict[str, object]]:
    if disjoint:
        return None, empty_error_profile()
    winner = choose_winner(aggregates)
    profile_pairs = decision_pairs.get(winner, []) if winner else []
    return winner, error_profile(profile_pairs)



# What: Scores both models against the typed transcripts.
# Needs: Directories for transcripts, predictions, and page photos.
# Returns: Aggregates, the winner, the error profile, disagreements, and report excerpts.
def compare(
    truth_dir: Path,
    predictions_dir: Path,
    pages_dir: Path,
) -> dict[str, object]:
    predictions = load_predictions(predictions_dir)
    truth_stems, scored_stems, pages, per_model_pairs = score_pages(truth_dir, predictions)
    aggregates, decision_pairs, decision_stems, disjoint, shared = align_models(
        pages,
        per_model_pairs,
        predictions,
        scored_stems,
    )
    winner, profile = judge(aggregates, decision_pairs, disjoint)
    disagreements = disagreeing_stems(decision_stems, pages)
    both_finished = all(aggregates[slug]["pages"] for slug in MODEL_SLUGS)
    page_stems = set(list_stems(pages_dir, {".jpg", ".jpeg", ".png"}))
    return {
        "pages": pages,
        "aggregates": aggregates,
        "winner": winner,
        "error_profile": profile,
        "disagreements": disagreements,
        "side_by_side": pages_to_show(decision_stems, pages, winner, disagreements),
        "skipped": sorted(page_stems - set(truth_stems)),
        "compared_pages": len(shared) if both_finished and shared else None,
        "disjoint": disjoint,
    }
