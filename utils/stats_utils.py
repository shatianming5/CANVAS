from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def _safe_group_counts(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    if x not in df.columns or y not in df.columns:
        return pd.Series(dtype=float)
    working = df[[x, y]].dropna(subset=[x])
    if working.empty:
        return pd.Series(dtype=float)
    return working.groupby(x)[y].count()


def annotate_sample_size(ax: Axes, df: pd.DataFrame, x: str, y: str, *, fontsize: int = 9) -> None:
    """Annotate per-group sample sizes (n=) above the primary axis ticks."""
    if not isinstance(ax, Axes):
        return
    counts = _safe_group_counts(df, x, y)
    if counts.empty:
        return

    xticks = list(ax.get_xticks())
    if len(xticks) != len(counts):
        return

    y_low, y_high = ax.get_ylim()
    span = max(y_high - y_low, 1e-6)
    baseline = y_high + span * 0.04

    for idx, (label, count) in enumerate(counts.items()):
        if idx >= len(xticks):
            break
        if count <= 0:
            continue
        ax.text(
            xticks[idx],
            baseline,
            f"n={int(count)}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#444444",
            clip_on=False,
        )

    ax.set_ylim(y_low, baseline + span * 0.12)


def _compute_ci(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    mean = float(np.nanmean(arr))
    if arr.size < 2:
        return (mean, 0.0)
    std = float(np.nanstd(arr, ddof=1))
    z = 1.96 if math.isclose(confidence, 0.95, rel_tol=1e-3) else 1.96
    ci = z * (std / math.sqrt(arr.size))
    if math.isnan(ci):
        ci = 0.0
    return (mean, ci)


def add_errorbars(
    ax: Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    confidence: float = 0.95,
    color: str = "#333333",
) -> None:
    """Add 95% CI error bars for aggregated points/bars."""
    if not isinstance(ax, Axes):
        return
    if x not in df.columns or y not in df.columns:
        return
    if not pd.api.types.is_numeric_dtype(df[y]):
        return

    grouped = df[[x, y]].dropna()
    if grouped.empty:
        return

    aggregates = grouped.groupby(x)[y].apply(list)
    if aggregates.empty:
        return

    xticks = list(ax.get_xticks())
    if len(xticks) != len(aggregates):
        return

    means = []
    errs = []
    positions = []
    for idx, (label, values) in enumerate(aggregates.items()):
        if idx >= len(xticks):
            break
        mean, ci = _compute_ci(values, confidence=confidence)
        if math.isnan(mean):
            continue
        means.append(mean)
        errs.append(ci)
        positions.append(xticks[idx])

    if not positions:
        return

    ax.errorbar(
        positions,
        means,
        yerr=errs,
        fmt="none",
        ecolor=color,
        elinewidth=1.2,
        capsize=3,
        alpha=0.85,
        zorder=10,
    )
