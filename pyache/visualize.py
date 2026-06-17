"""Visualization helpers for time-series explanations."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_signal_with_importance(signal: np.ndarray, importance: np.ndarray, outpath: str | Path, title: str | None = None) -> None:
    """Save a plot overlaying importance heatmap over the 1D signal.

    Args:
        signal: 1D array (T,) or shape (T, C) will take first channel.
        importance: 1D array (T,) in [0,1].
        outpath: file path to save PNG.
    """
    sig = np.asarray(signal)
    if sig.ndim == 2:
        sig = sig[:, 0]
    imp = np.asarray(importance)
    if imp.shape[0] != sig.shape[0]:
        raise ValueError("importance and signal must have same length")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(sig, color="#222222", lw=1)
    ax.set_xlim(0, len(sig) - 1)
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Sample")
    if title:
        ax.set_title(title)

    # color the background by importance
    cmap = plt.get_cmap("Reds")
    for i in range(len(sig) - 1):
        a = imp[i]
        ax.axvspan(i, i + 1, color=cmap(a), alpha=0.5 * a)

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150)
    plt.close(fig)
