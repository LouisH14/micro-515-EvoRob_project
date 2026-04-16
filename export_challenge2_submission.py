"""Export the Challenge 2 submission artifacts from an NSGA-II checkpoint.

This script extracts three controllers from a completed multi-objective run:
- Flat specialist: best score on objective 1
- Ice specialist: best score on objective 2
- Generalist: a balanced solution from the first Pareto front

It creates checkpoint-style folders that contain the expected `.npy` files so
the selected controllers can be replayed or evaluated independently.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from evorob.algorithms.nsga import NSGAII


def load_last_generation(checkpoint_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the final generation fitness and population arrays."""
    full_f = np.load(checkpoint_dir / "full_f.npy", allow_pickle=True)
    full_x = np.load(checkpoint_dir / "full_x.npy", allow_pickle=True)

    if full_f.ndim != 3 or full_x.ndim != 3:
        raise ValueError(
            "Expected full_f.npy and full_x.npy to have shape (generations, pop, ...)"
        )

    return full_f[-1], full_x[-1]


def select_controllers(
    fitness: np.ndarray, population: np.ndarray
) -> Dict[str, Dict[str, np.ndarray | int]]:
    """Select flat specialist, ice specialist, and a balanced generalist."""
    nsga = NSGAII(population_size=fitness.shape[0], n_opt_params=population.shape[1])
    fronts, _ = nsga.fast_nondominated_sort(fitness)

    flat_idx = int(np.argmax(fitness[:, 0]))
    ice_idx = int(np.argmax(fitness[:, 1]))

    front0 = fronts[0] if fronts else list(range(len(fitness)))
    front0_fitness = fitness[front0]
    front_min = front0_fitness.min(axis=0)
    front_max = front0_fitness.max(axis=0)
    span = np.where(front_max - front_min == 0, 1.0, front_max - front_min)
    normalized = (front0_fitness - front_min) / span
    ideal = np.ones(normalized.shape[1])
    generalist_pos = int(np.argmin(np.linalg.norm(ideal - normalized, axis=1)))
    generalist_idx = int(front0[generalist_pos])

    return {
        "flat_specialist": {
            "index": flat_idx,
            "genotype": population[flat_idx],
            "fitness": fitness[flat_idx],
        },
        "ice_specialist": {
            "index": ice_idx,
            "genotype": population[ice_idx],
            "fitness": fitness[ice_idx],
        },
        "generalist": {
            "index": generalist_idx,
            "genotype": population[generalist_idx],
            "fitness": fitness[generalist_idx],
        },
    }


def save_checkpoint_like(
    target_dir: Path,
    genotype: np.ndarray,
    fitness: np.ndarray,
) -> None:
    """Save a single controller in the same file layout as an EA checkpoint."""
    target_dir.mkdir(parents=True, exist_ok=True)

    genotype = np.asarray(genotype)
    fitness = np.asarray(fitness)

    np.save(target_dir / "x.npy", genotype[np.newaxis, :])
    np.save(target_dir / "f.npy", fitness[np.newaxis, :])
    np.save(target_dir / "x_best.npy", genotype)
    np.save(target_dir / "f_best.npy", fitness)
    np.save(target_dir / "controller.npy", genotype)


def export_submission(checkpoint_dir: Path, output_dir: Path) -> None:
    """Create the submission artifacts for Challenge 2."""
    fitness, population = load_last_generation(checkpoint_dir)
    selected = select_controllers(fitness, population)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    bundle_payload = {}

    for label, payload in selected.items():
        folder = output_dir / f"{label}_checkpoint"
        save_checkpoint_like(folder, payload["genotype"], payload["fitness"])

        bundle_payload[f"{label}_genotype"] = np.asarray(payload["genotype"])
        bundle_payload[f"{label}_fitness"] = np.asarray(payload["fitness"])

        fitness_str = np.array2string(np.asarray(payload["fitness"]), precision=6)
        manifest_lines.append(
            f"{label}: index={payload['index']} fitness={fitness_str} folder={folder.name}"
        )

    np.savez(output_dir / "controller_bundle.npz", **bundle_payload)

    manifest_path = output_dir / "selected_controllers.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    metadata = {
        "source_checkpoint": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "selected": {
            label: {
                "index": int(payload["index"]),
                "fitness": np.asarray(payload["fitness"]).tolist(),
            }
            for label, payload in selected.items()
        },
    }
    (output_dir / "selected_controllers.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for artifact in ("pareto_fronts.pdf", "fitness_plot.pdf", "evaluation_score.txt"):
        src = checkpoint_dir / artifact
        if src.exists():
            shutil.copy2(src, output_dir / artifact)

    print(f"Exported submission artifacts to: {output_dir}")
    for line in manifest_lines:
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Challenge 2 submission artifacts from a checkpoint."
    )
    parser.add_argument(
        "checkpoint_dir",
        nargs="?",
        default="results/20260402_143145_nsga_ckpts",
        help="Path to the NSGA-II checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where the extracted controllers will be written. "
            "Defaults to <checkpoint_dir>/submission_export."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_dir / "submission_export"
    )
    export_submission(checkpoint_dir, output_dir)


if __name__ == "__main__":
    main()
