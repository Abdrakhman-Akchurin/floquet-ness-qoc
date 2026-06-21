"""Minimal plotting utility for saved NPZ runs.

This script is intentionally small. For publication/presentation figures, make a copy
and adjust styling rather than mixing plotting code back into optimization scripts.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REFERENCE_DIR = Path("results/reference_runs")
FIG_DIR = Path("results/generated_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_bell_controls(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    if not {"s_nodes", "delta_s", "gamma_s"}.issubset(data.files):
        return
    s = data["s_nodes"]
    fig, ax1 = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    ax1.plot(s, data["delta_s"], label=r"$\delta(s)$")
    ax1.plot(s, data["gamma_s"], label=r"$\gamma(s)$")
    ax1.set_xlabel(r"normalized time $s=t/T$")
    ax1.set_ylabel("control amplitude")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{path.stem}_controls.png")
    plt.close(fig)


def plot_single_history(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    candidates = ["hist_J", "history_F", "history_T"]
    keys = [key for key in candidates if key in data.files]
    for key in keys:
        arr = np.asarray(data[key], dtype=float)
        if arr.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
        ax.plot(np.arange(arr.size), arr)
        ax.set_xlabel("optimizer callback index")
        ax.set_ylabel(key)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{path.stem}_{key}.png")
        plt.close(fig)


if __name__ == "__main__":
    for path in sorted(REFERENCE_DIR.glob("*.npz")):
        plot_bell_controls(path)
        plot_single_history(path)
    print(f"Saved generated figures to {FIG_DIR}")
