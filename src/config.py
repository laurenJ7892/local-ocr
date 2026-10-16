"""Paths, model ids, and scoring thresholds shared by the scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "data" / "pages"
TRUTH_DIR = ROOT / "data" / "ground_truth"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
RESIZED_DIR = ROOT / "outputs" / "resized"
PID_PATH = ROOT / "outputs" / "transcribe.pid"
REPORT_PATH = ROOT / "outputs" / "report.md"
SCORES_PATH = ROOT / "outputs" / "scores.json"

LONG_EDGE = 1600
MAX_TOKENS = 2048
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXCERPT_LIMIT = 1200
SIDE_BY_SIDE_LIMIT = 8

CER_GOOD_ENOUGH = 0.08
CER_FINETUNE_PLAUSIBLE = 0.20
CLOSE_CER = 0.02

PROMPT = (
    "Transcribe this handwritten notebook page exactly. "
    "Keep line breaks. Leave spelling as written. "
    "Mark unreadable words as [illegible]. Do not add commentary."
)

MODELS = (
    {"slug": "glm-ocr", "repo": "mlx-community/GLM-OCR-bf16"},
    {"slug": "qwen3.5-9b", "repo": "mlx-community/Qwen3.5-9B-MLX-4bit"},
)

MODEL_SLUGS = tuple(spec["slug"] for spec in MODELS)
