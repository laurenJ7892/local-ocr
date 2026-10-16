# Local handwriting comparison

This project transcribes photos of notebook pages with two open vision models and scores them against transcripts you type. Both models run on this Mac through [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). The weights download from Hugging Face on the first run and stay in the local cache.

The models are:

- `mlx-community/GLM-OCR-bf16` (0.9B, MIT), saved under `outputs/predictions/glm-ocr/`
- `mlx-community/Qwen3.5-9B-MLX-4bit` (Apache 2.0), saved under `outputs/predictions/qwen3.5-9b/`

A Python script loads one model, reads the photos, and writes a text file per page. There is no Docker container and no API server. When the last page is written, the process exits and the model leaves memory.

## Project structure

```text
data/pages/              notebook photos
data/ground_truth/       the transcript you type for each photo
outputs/                 predictions, resized photos, the report, and the scores
src/config.py            shared paths, model ids, and scoring thresholds
src/errors.py            exit codes and the messages the scripts print
src/transcribe.py        loads a model and writes one text file per page
src/stop.py              interrupts a transcription that is still running
src/text.py              strict and normalized views of a transcription
src/scoring.py           character and word error rates, and which model wins
src/evaluate.py          the fine-tune decision, the markdown report, and the score command
tests/test_text.py       tests for the text views
tests/test_scoring.py    tests for the rates, the decision bands, and page helpers
pyproject.toml           the Python version and the project dependencies
```

## Setup

The environment uses Python 3.12 because MLX does not provide wheels for this machine's system Python.

```bash
uv sync
uv run ruff check src tests
```

## Add pages

Put one photo per page in `data/pages/`. Use `.jpg` or `.png`. For each photo, type what the page actually says, including your spelling and abbreviations, and save it as `data/ground_truth/<same-name>.txt`.

```text
data/pages/page01.jpg
data/ground_truth/page01.txt
```

A useful first set is 20–30 pages that cover neat and rushed writing, pen and pencil, and full pages and short notes. Pages without a transcript are transcribed, and the scorer skips them.

## Transcribe

```bash
uv run python src/transcribe.py
```

The script resizes each photo so the long edge is at most 1600 pixels, then asks the model to copy the handwriting exactly, keep line breaks, leave spelling as written, and mark unreadable words as `[illegible]`. It writes each finished page immediately. A blank result is not saved, and the next run tries that page again. A later run skips a page whose prediction contains text and is newer than the photo. Replacing the photo causes that page to be transcribed again.

Run one model with:

```bash
uv run python src/transcribe.py --model glm-ocr
uv run python src/transcribe.py --model qwen3.5-9b
```

## Stop a run

Press Ctrl+C in the terminal that is transcribing. The current page is discarded, finished pages stay on disk, and the next run continues with the rest. Press Ctrl+C a second time to stop immediately.

From another terminal:

```bash
uv run python src/stop.py
```

That sends the same interrupt. Running it again while the process is still in a page stops it immediately.

## Score

```bash
uv run python src/evaluate.py
```

This writes `outputs/report.md` and `outputs/scores.json`.

Character error rate (CER) is the share of characters inserted, deleted, or changed. Word error rate (WER) is the same idea for words. CER is the number to use for handwriting. Each rate is reported twice:

- Strict: the text as written, aside from Windows line endings.
- Normalized: case, extra spaces, and curly quotes are folded. Spelling stays as written, so a model that “corrects” a word still counts as wrong.

The report names the model with the lower normalized CER, lists pages where the two transcriptions differ, and shows the transcript beside each prediction for the weakest pages.

The fine-tune line uses the better model:

- Normalized CER at or below 8%, with misses mostly punctuation or line breaks: that model is good enough.
- Normalized CER from 8% to 20%: fine-tuning is plausible. If the errors cluster on a few letter shapes, the report says so. GLM-OCR is the cheaper model to adapt when it is close, because it is under 1B parameters.
- Normalized CER above 20%: fine-tuning on your handwriting is justified, starting from the winning model.

The scripts do not train anything.
