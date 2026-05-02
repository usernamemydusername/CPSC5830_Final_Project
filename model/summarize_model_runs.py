#!/usr/bin/env python3
"""
Summarize test metrics across multiple seeds and multiple models.

Expected run layout, for each model:

  MODEL_RUN_DIR/
  ├── seed123/test_metrics.json
  ├── seed456/test_metrics.json
  └── seed789/test_metrics.json

It also supports a single-run directory with:

  MODEL_RUN_DIR/test_metrics.json

Example:

  python summarize_model_runs.py \
    --model homo_gru=/nfs/.../runs/homo_gru_baseline \
    --model hetero_rgcn_gru=/nfs/.../runs/hetero_rgcn_gru/full \
    --seeds 123 456 789 \
    --out-dir /nfs/.../runs/summary

Outputs:
  - all_seed_metrics.csv: one row per model per seed
  - summary_metrics.csv: mean/std/count for every numeric metric
  - summary_metrics.json: same information in JSON form

This file is intentionally independent of torch / torch_geometric so it can run
quickly after training jobs finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_model_arg(text: str) -> Tuple[str, Path]:
    """Parse NAME=PATH."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"Invalid --model value {text!r}. Expected format NAME=/path/to/run_dir"
        )
    name, path = text.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError("Model name cannot be empty.")
    if not path:
        raise argparse.ArgumentTypeError("Model path cannot be empty.")
    return name, Path(path)


def is_number(x) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_seed_metrics(model_name: str, run_dir: Path, seeds: Optional[List[int]]) -> List[dict]:
    """
    Read test_metrics.json for one model.

    Priority:
      1. If seeds are provided, look for seed{seed}/test_metrics.json.
      2. If no seeds are provided, discover seed*/test_metrics.json.
      3. If no seed directories are found, fall back to run_dir/test_metrics.json.
    """
    rows: List[dict] = []

    if seeds:
        for seed in seeds:
            path = run_dir / f"seed{seed}" / "test_metrics.json"
            if not path.exists():
                print(f"[WARN] Missing metrics for {model_name}, seed {seed}: {path}")
                continue
            metrics = read_json(path)
            rows.append({"model": model_name, "seed": seed, **metrics})
        return rows

    discovered = sorted(run_dir.glob("seed*/test_metrics.json"))
    if discovered:
        for path in discovered:
            seed_text = path.parent.name.replace("seed", "")
            seed = int(seed_text) if seed_text.isdigit() else seed_text
            metrics = read_json(path)
            rows.append({"model": model_name, "seed": seed, **metrics})
        return rows

    single = run_dir / "test_metrics.json"
    if single.exists():
        metrics = read_json(single)
        rows.append({"model": model_name, "seed": "single", **metrics})
        return rows

    print(f"[WARN] No test_metrics.json found for {model_name} under {run_dir}")
    return rows


def summarize_rows(rows: List[dict]) -> List[dict]:
    """Return one summary row per model with metric_mean / metric_std columns."""
    by_model: Dict[str, List[dict]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    summary_rows: List[dict] = []
    for model_name, model_rows in sorted(by_model.items()):
        metric_keys = sorted({
            key
            for row in model_rows
            for key, val in row.items()
            if key not in {"model", "seed"} and is_number(val)
        })
        out = {"model": model_name, "num_runs": len(model_rows)}
        for key in metric_keys:
            vals = [float(row[key]) for row in model_rows if key in row and is_number(row[key])]
            if not vals:
                continue
            out[f"{key}_mean"] = sum(vals) / len(vals)
            out[f"{key}_std"] = stats.stdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        print(f"[WARN] No rows to write for {path}")
        return
    fieldnames = ["model", "seed"] if "seed" in rows[0] else ["model"]
    remaining = sorted({k for row in rows for k in row.keys()} - set(fieldnames))
    fieldnames = fieldnames + remaining
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_arg,
        required=True,
        help="Model run directory in NAME=/path/to/run_dir format. Can be repeated.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Optional seed list. If omitted, seed*/ folders are auto-discovered.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where summary CSV/JSON files will be written.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    for model_name, run_dir in args.model:
        rows = find_seed_metrics(model_name, run_dir, args.seeds)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No metrics were found. Check --model paths and seed folders.")

    summary_rows = summarize_rows(all_rows)

    all_csv = args.out_dir / "all_seed_metrics.csv"
    summary_csv = args.out_dir / "summary_metrics.csv"
    summary_json = args.out_dir / "summary_metrics.json"

    write_csv(all_csv, all_rows)
    write_csv(summary_csv, summary_rows)

    payload = {
        "models": [name for name, _ in args.model],
        "seeds": args.seeds,
        "all_seed_metrics": all_rows,
        "summary": summary_rows,
    }
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("Wrote summary files:")
    print(f"  {all_csv}")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")

    print("\nCompact summary:")
    for row in summary_rows:
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
