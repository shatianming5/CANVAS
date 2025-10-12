from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = Path(
    "cleaned_41586_2013_BFnature12437_MOESM8_ESM_Figure_1d_tidy_preview.csv"
)
OUTPUT_PATH = Path("outputs/figure1d_caption.png")


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def prepare_extracellular_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby(["genotype", "condition", "metabolite"])
        .agg(
            mean_ratio=("ratio_metabolite_glucose_area_1e6_cells", "mean"),
            std_ratio=("ratio_metabolite_glucose_area_1e6_cells", "std"),
            n=("ratio_metabolite_glucose_area_1e6_cells", "count"),
        )
        .reset_index()
    )
    return stats


def prepare_intracellular_stats(df: pd.DataFrame) -> pd.DataFrame:
    inset_df = df.dropna(subset=["metabolite_2", "condition_2", "x13c_area_1e6_cells"])
    stats = (
        inset_df.groupby(["metabolite_2", "condition_2"])
        .agg(
            mean_level=("x13c_area_1e6_cells", "mean"),
            std_level=("x13c_area_1e6_cells", "std"),
        )
        .reset_index()
    )
    return stats


def plot_panel_d(ax: plt.Axes, stats: pd.DataFrame) -> None:
    genotypes = sorted(stats["genotype"].unique())
    conditions = ["untreated", "ADR"]
    combo_labels = [f"{g}\n{cond}" for g in genotypes for cond in conditions]
    x_positions = np.arange(len(combo_labels))
    metabolites = stats["metabolite"].unique()

    width = 0.35 if len(metabolites) == 2 else 0.25
    palette = sns.color_palette("Set2", len(metabolites))

    for idx, metabolite in enumerate(metabolites):
        offsets = x_positions + (idx - (len(metabolites) - 1) / 2) * width
        means = []
        errors = []
        nd_flags = []
        for g in genotypes:
            for cond in conditions:
                row = stats[
                    (stats["genotype"] == g)
                    & (stats["condition"] == cond)
                    & (stats["metabolite"] == metabolite)
                ]
                if row.empty:
                    means.append(0.0)
                    errors.append(0.0)
                    nd_flags.append(True)
                else:
                    mean_val = row["mean_ratio"].iloc[0]
                    std_val = row["std_ratio"].iloc[0]
                    means.append(mean_val)
                    errors.append(std_val if pd.notna(std_val) else 0.0)
                    nd_flags.append(abs(mean_val) < 1e-4)
        bars = ax.bar(
            offsets,
            means,
            width=width,
            color=palette[idx],
            label=metabolite,
            edgecolor="black",
            linewidth=0.6,
            yerr=errors,
            capsize=4,
        )
        for bar, is_nd in zip(bars, nd_flags):
            if is_nd:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    "n.d.",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(combo_labels)
    ax.set_ylabel("Extracellular metabolite/glucose ratio")
    ax.set_title("Panel d: Extracellular metabolites (mean ± s.d., n=4)")
    ax.legend(title="Metabolite", frameon=False)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(bottom=0)
    ax.annotate(
        "Glucose-6-phosphate not detectable in medium",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
    )


def plot_inset(ax: plt.Axes, inset_stats: pd.DataFrame) -> None:
    metabolites = inset_stats["metabolite_2"].unique()
    conditions = ["untreated", "ADR"]
    x_positions = np.arange(len(metabolites))
    width = 0.35
    palette = {"untreated": sns.color_palette("Set1")[2], "ADR": sns.color_palette("Set1")[0]}

    for idx, cond in enumerate(conditions):
        offsets = x_positions + (idx - 0.5) * width
        means = []
        errors = []
        for metabolite in metabolites:
            row = inset_stats[
                (inset_stats["metabolite_2"] == metabolite)
                & (inset_stats["condition_2"] == cond)
            ]
            if row.empty:
                means.append(0.0)
                errors.append(0.0)
            else:
                means.append(row["mean_level"].iloc[0])
                std_val = row["std_level"].iloc[0]
                errors.append(std_val if pd.notna(std_val) else 0.0)
        ax.bar(
            offsets,
            means,
            width=width,
            color=palette.get(cond, "gray"),
            label=cond,
            yerr=errors,
            capsize=3,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([m.replace("#", "") for m in metabolites], rotation=45, ha="right")
    ax.set_ylabel("Intracellular level (a.u.)")
    ax.set_title("Panel inset: ADR vs untreated (13C6-glucose labeling)")
    ax.legend(title="Condition", frameon=False, fontsize=8)


def main() -> None:
    df = load_data()
    ext_stats = prepare_extracellular_stats(df)
    inset_stats = prepare_intracellular_stats(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    plot_panel_d(axes[0], ext_stats)
    plot_inset(axes[1], inset_stats)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
