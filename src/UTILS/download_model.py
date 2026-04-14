"""
download_model.py — Download any HuggingFace model given a model_id and output dir.

Usage:
    python download_model.py --model_id google/gemma-3-1b-it --model_dir ./models/gemma-3-1b-it
    python download_model.py --model_id mistralai/Mistral-7B-v0.1  # uses $DIRWORK_FTE env var as base
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HFValidationError

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "flax_model*", "tf_model*"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def resolve_model_dir(model_id: str, model_dir: str | None) -> Path:
    """
    Priority:
      1. Explicit --model_dir argument
      2. ./models/<model_slug>
    """
    slug = model_id.split("/")[-1]

    if model_dir:
        return Path(model_dir)

    return Path("models") / slug


def validate_model_id(model_id: str) -> None:
    """Check the repo exists on HuggingFace before downloading."""
    try:
        api = HfApi()
        api.repo_info(repo_id=model_id, repo_type="model")
    except RepositoryNotFoundError:
        log.error(f"Model '{model_id}' not found on HuggingFace Hub.")
        sys.exit(1)
    except HFValidationError as e:
        log.error(f"Invalid model id '{model_id}': {e}")
        sys.exit(1)


def download_model(model_id: str, model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Downloading '{model_id}' → {model_dir}")
    log.info(f"Ignoring patterns: {IGNORE_PATTERNS}")

    path = snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        ignore_patterns=IGNORE_PATTERNS,
    )
    return Path(path)


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model snapshot locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID, e.g. 'google/gemma-3-1b-it'",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=(
            "Local directory to save the model. "
            "Defaults to $DIRWORK_FTE/models/<model_slug> or ./models/<model_slug>."
        ),
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip HuggingFace Hub existence check before downloading.",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    model_dir = resolve_model_dir(args.model_id, args.model_dir)

    log.info(f"Model ID  : {args.model_id}")
    log.info(f"Model dir : {model_dir}")

    if not args.skip_validation:
        log.info("Validating model ID on HuggingFace Hub...")
        validate_model_id(args.model_id)

    path = download_model(args.model_id, model_dir)

    log.info(f"✓ Model downloaded to: {path}")


if __name__ == "__main__":
    main()