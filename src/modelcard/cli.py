"""Command-line interface for modelcard.

Two subcommands:

* ``modelcard demo`` — train the bundled Iris demo model and emit its card.
* ``modelcard render`` — load a pickled estimator plus a JSON spec describing
  metrics / dataset / model, and emit the corresponding card.

Both write to stdout by default, or to ``--out PATH`` when given.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

from .card import render_model_card
from .demo import render_demo_card
from .metadata import DatasetInfo, ModelInfo


def _load_spec(path: Path) -> tuple[dict[str, Any], DatasetInfo, ModelInfo]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("spec file must contain a JSON object")

    for key in ("metrics", "dataset", "model"):
        if key not in raw:
            raise ValueError(f"spec is missing required key '{key}'")

    metrics = raw["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("'metrics' must be a JSON object")

    dataset = DatasetInfo(**raw["dataset"])
    model = ModelInfo(**raw["model"])
    return metrics, dataset, model


def _write(text: str, out: str | None) -> None:
    if out is None:
        sys.stdout.write(text)
    else:
        Path(out).write_text(text, encoding="utf-8")


def _cmd_demo(args: argparse.Namespace) -> int:
    card = render_demo_card(top_k_features=args.top_k)
    _write(card, args.out)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    with Path(args.estimator).open("rb") as fh:
        estimator = pickle.load(fh)
    metrics, dataset, model = _load_spec(Path(args.spec))
    card = render_model_card(
        estimator, metrics, dataset, model, top_k_features=args.top_k
    )
    _write(card, args.out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (exposed for testing)."""
    parser = argparse.ArgumentParser(
        prog="modelcard",
        description="Generate a Markdown model card from a scikit-learn estimator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="train the bundled Iris demo and emit its card")
    demo.add_argument("--out", default=None, help="output file (default: stdout)")
    demo.add_argument(
        "--top-k", type=int, default=10, help="max features in importance table"
    )
    demo.set_defaults(func=_cmd_demo)

    render = sub.add_parser("render", help="render a card from a pickled estimator + spec")
    render.add_argument("--estimator", required=True, help="path to a pickled estimator")
    render.add_argument("--spec", required=True, help="path to a JSON spec file")
    render.add_argument("--out", default=None, help="output file (default: stdout)")
    render.add_argument(
        "--top-k", type=int, default=10, help="max features in importance table"
    )
    render.set_defaults(func=_cmd_render)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
    except (ValueError, FileNotFoundError, TypeError) as exc:
        parser.error(str(exc))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
