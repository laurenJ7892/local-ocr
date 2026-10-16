"""Transcribe notebook photos with local Hugging Face vision models."""

from __future__ import annotations

import argparse
import gc
import os
import signal
from pathlib import Path

from PIL import Image

from config import (
    IMAGE_SUFFIXES,
    LONG_EDGE,
    MAX_TOKENS,
    MODELS,
    PAGES_DIR,
    PID_PATH,
    PREDICTIONS_DIR,
    PROMPT,
    RESIZED_DIR,
    ROOT,
)
from errors import (
    EXIT_OK,
    EXIT_STOPPED,
    FINISHED,
    STOP_REQUESTED,
    STOPPED,
    STOPPING_IMMEDIATELY,
    error,
    guard,
    info,
    write_output,
)


class StopFlag:
    """First interrupt finishes between pages. The second exits immediately."""

    def __init__(self) -> None:
        self.count = 0

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, frame: object) -> None:
        self.count += 1
        if self.count >= 2:
            error(STOPPING_IMMEDIATELY)
            raise SystemExit(EXIT_STOPPED)
        error(STOP_REQUESTED)

    @property
    def requested(self) -> bool:
        return self.count > 0


def list_pages(pages_dir: Path) -> list[Path]:
    if not pages_dir.is_dir():
        return []
    return sorted(
        path
        for path in pages_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def resize_long_edge(source: Path, dest: Path, long_edge: int = LONG_EDGE) -> Path:
    """Write a JPEG whose longer side is at most `long_edge` pixels."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > long_edge:
            scale = long_edge / longest
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        image.save(dest, format="JPEG", quality=90)
    return dest


def prediction_path(predictions_dir: Path, slug: str, page: Path) -> Path:
    return predictions_dir / slug / f"{page.stem}.txt"


def already_done(path: Path, source: Path | None = None) -> bool:
    """A prediction is current when it is non-empty and newer than the photo."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if source is not None and source.stat().st_mtime > path.stat().st_mtime:
        return False
    return True


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_pid(pid_path: Path) -> None:
    write_output(pid_path, f"{os.getpid()}\n")


def clear_pid(pid_path: Path) -> None:
    try:
        recorded = pid_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return
    if recorded == str(os.getpid()):
        pid_path.unlink()


def release_model(model: object, processor: object) -> None:
    del model, processor
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def transcribe_page(model: object, processor: object, config: object, image_path: Path) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted = apply_chat_template(
        processor,
        config,
        PROMPT,
        num_images=1,
    )
    result = generate(
        model,
        processor,
        formatted,
        image=[str(image_path)],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        enable_thinking=False,
    )
    text = result.text if hasattr(result, "text") else str(result)
    return text.strip()


def run_model(
    spec: dict[str, str],
    pages: list[Path],
    predictions_dir: Path,
    resized_dir: Path,
    stop: StopFlag,
) -> bool:
    """Transcribe every page with one model. Returns False when stopped early."""
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    pending = [
        page
        for page in pages
        if not already_done(prediction_path(predictions_dir, spec["slug"], page), page)
    ]
    if not pending:
        info(f"{spec['slug']}: every page already has a prediction.")
        return True
    if stop.requested:
        return False

    info(f"Loading {spec['repo']}")
    model, processor = load(spec["repo"])
    config = load_config(spec["repo"])
    try:
        for index, page in enumerate(pending, start=1):
            if stop.requested:
                return False
            started = stop.count
            dest = prediction_path(predictions_dir, spec["slug"], page)
            resized = resized_dir / f"{page.stem}.jpg"
            info(f"{spec['slug']}: {index}/{len(pending)} {page.name}")
            resize_long_edge(page, resized)
            text = transcribe_page(model, processor, config, resized)
            if stop.count != started:
                info(f"{spec['slug']}: discarded {page.name}")
                return False
            write_output(dest, text + "\n")
            info(f"{spec['slug']}: wrote {display_path(dest)}")
    finally:
        release_model(model, processor)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe notebook pages locally.")
    parser.add_argument("--pages", type=Path, default=PAGES_DIR)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--resized", type=Path, default=RESIZED_DIR)
    parser.add_argument("--pid-file", type=Path, default=PID_PATH)
    parser.add_argument(
        "--model",
        action="append",
        choices=[spec["slug"] for spec in MODELS],
        help="Run one model. Repeat to select several. Default: both.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pages = list_pages(args.pages)
    if not pages:
        info(f"No photos in {args.pages}. Add .jpg or .png files and run again.")
        return EXIT_OK

    selected = [spec for spec in MODELS if args.model is None or spec["slug"] in args.model]
    stop = StopFlag()
    stop.install()
    write_pid(args.pid_file)
    stopped = False
    try:
        for spec in selected:
            if not run_model(spec, pages, args.predictions, args.resized, stop):
                stopped = True
                break
    finally:
        clear_pid(args.pid_file)
    if stopped:
        info(STOPPED)
        return EXIT_STOPPED
    info(FINISHED)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(guard(main))
