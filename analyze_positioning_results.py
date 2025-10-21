"""Generate diagnostic plots from positioning_sweep_results.csv."""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ARRAY_COLUMNS = [
    "MaxCorrSeries",
    "TDOAMeters",
    "TDOASeconds",
    "PropagationLossSeries",
    "PropagationDelaySeries",
    "AvailableMask",
    "IsLOSMask",
    "SelectedAnchors",
    "BaselineAnchors",
    "CandidateAnchors",
    "RansacInliers",
    "ResidualSeries",
    "CoarseEstimate",
    "CoarseCovariance",
    "DetectedMask",
    "TxPositions",
]


def _parse_matlab_like_array(text: str) -> np.ndarray:
    """Parse MATLAB-style bracket arrays into numpy arrays."""
    text = text.strip()
    if not text:
        return np.array([], dtype=float)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return np.array([], dtype=float)
    rows = [row.strip() for row in text.split(";") if row.strip()]
    if not rows:
        return np.array([], dtype=float)
    data: list[list[float]] = []
    for row in rows:
        tokens = row.replace(",", " ").split()
        if not tokens:
            continue
        data.append([float(tok) for tok in tokens])
    if not data:
        return np.array([], dtype=float)
    arr = np.array(data, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.flatten()
    return arr


def parse_array(value: str | float | None) -> np.ndarray:
    """Convert a stringified array (MATLAB-style) to numpy array."""
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, float) and np.isnan(value):
        return np.array([], dtype=float)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return np.array([], dtype=float)
        try:
            return _parse_matlab_like_array(value)
        except ValueError:
            return np.array([], dtype=float)
    return np.array([], dtype=float)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results CSV and expand derived columns."""
    df = pd.read_csv(csv_path)
    for col in ARRAY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_array)

    df["ScenarioLabel"] = df.apply(
        lambda row: f"SNR {row['ConfiguredSNR_dB']} dB - UKF {'On' if bool(row['UKFEnabled']) else 'Off'}",
        axis=1,
    )
    df["HorizontalError"] = df["HorizontalError"].astype(float)
    df["MaxCorrStrongest"] = df["MaxCorrStrongest"].astype(float)
    df["ScenarioLabel"] = df["ScenarioLabel"].astype("category")
    return df


def ensure_output_dir(output_dir: Path) -> None:
    """Create output directory if it does not exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


def plot_position_error_cdf(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot horizontal error empirical CDF per scenario."""
    plt.figure(figsize=(7, 5))
    for label, group in df.groupby("ScenarioLabel"):
        errors = np.sort(group["HorizontalError"].dropna().values)
        if errors.size == 0:
            continue
        cdf = np.linspace(1.0 / errors.size, 1.0, errors.size)
        plt.step(errors, cdf, where="post", label=label)
    plt.xlabel("Horizontal Error (m)")
    plt.ylabel("Empirical CDF")
    plt.title("Horizontal Position Error CDF")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "position_error_cdf.png", dpi=300)
    plt.close()


def plot_error_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean and 95th percentile horizontal error per scenario."""
    summary = (
        df.groupby("ScenarioLabel")["HorizontalError"]
        .agg(MeanError="mean", Error95=lambda s: np.percentile(s.dropna(), 95))
        .reset_index()
    )
    summary = summary.melt(id_vars="ScenarioLabel", var_name="Metric", value_name="Error(m)")
    plt.figure(figsize=(7, 4))
    sns.barplot(data=summary, x="ScenarioLabel", y="Error(m)", hue="Metric")
    plt.ylabel("Error (m)")
    plt.xlabel("")
    plt.title("Horizontal Error Summary by Scenario")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "position_error_summary.png", dpi=300)
    plt.close()


def plot_error_by_position(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate boxplots of horizontal error per UE position."""
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="PositionIndex", y="HorizontalError", hue="ScenarioLabel")
    plt.xlabel("Position Index")
    plt.ylabel("Horizontal Error (m)")
    plt.title("Horizontal Error Distribution per UE Position")
    plt.tight_layout()
    plt.savefig(output_dir / "horizontal_error_by_position.png", dpi=300)
    plt.close()


def plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot top-down scatter of horizontal error for each scenario."""
    scenario_labels = df["ScenarioLabel"].cat.categories
    n_cols = min(2, len(scenario_labels))
    n_rows = int(np.ceil(len(scenario_labels) / n_cols)) or 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    vmin = df["HorizontalError"].min()
    vmax = df["HorizontalError"].max()

    for ax, label in zip(axes.ravel(), scenario_labels):
        subset = df[df["ScenarioLabel"] == label]
        sc = ax.scatter(subset["TrueX"], subset["TrueY"], c=subset["HorizontalError"], cmap="viridis", vmin=vmin, vmax=vmax, s=80)
        ax.set_title(label)
        ax.set_xlabel("True X (m)")
        ax.set_ylabel("True Y (m)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_aspect("equal")

    for ax in axes.ravel()[len(scenario_labels) :]:
        ax.axis("off")

    fig.colorbar(sc, ax=axes, label="Horizontal Error (m)", shrink=0.8)
    fig.suptitle("Top-Down Position Error Map", y=0.95)
    fig.tight_layout()
    fig.savefig(output_dir / "horizontal_error_heatmap.png", dpi=300)
    plt.close(fig)


def plot_corr_vs_error(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter strongest correlation metric versus horizontal error."""
    plt.figure(figsize=(7, 4))
    sns.scatterplot(
        data=df,
        x="MaxCorrStrongest",
        y="HorizontalError",
        hue="ScenarioLabel",
        style="ScenarioLabel",
        s=70,
    )
    plt.xlabel("Strongest PRS Correlation Metric")
    plt.ylabel("Horizontal Error (m)")
    plt.title("Position Error vs. Detection Quality")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_correlation.png", dpi=300)
    plt.close()


def plot_corr_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of correlation metrics across anchors per scenario."""
    records = []
    for _, row in df.iterrows():
        series = row["MaxCorrSeries"]
        if series.size == 0:
            continue
        for value in series[np.isfinite(series) & (series > 0)]:
            records.append(
                {
                    "ScenarioLabel": row["ScenarioLabel"],
                    "CorrelationMetric": value,
                }
            )
    if not records:
        return
    corr_df = pd.DataFrame(records)
    plt.figure(figsize=(7, 4))
    sns.histplot(
        data=corr_df,
        x="CorrelationMetric",
        hue="ScenarioLabel",
        element="step",
        stat="density",
        common_norm=False,
        bins=20,
    )
    plt.xlabel("PRS Correlation Metric")
    plt.ylabel("Density")
    plt.title("Distribution of Anchor Detection Metrics")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_metric_histogram.png", dpi=300)
    plt.close()


def plot_solver_outcomes(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of centroid fallback rates per scenario."""
    fallback_summary = (
        df.groupby("ScenarioLabel")["CentroidFallback"]
        .agg(Total="count", Fallbacks="sum")
        .assign(FallbackRate=lambda g: g["Fallbacks"] / g["Total"].clip(lower=1))
        .reset_index()
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(data=fallback_summary, x="ScenarioLabel", y="FallbackRate")
    plt.ylabel("Fallback Rate")
    plt.xlabel("")
    plt.ylim(0, 1)
    plt.title("Centroid Fallback Rate by Scenario")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "centroid_fallback_rate.png", dpi=300)
    plt.close()


def main() -> None:
    """Entry-point for generating plots."""
    csv_path = Path("positioning_sweep_results.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found at {csv_path.resolve()}")

    df = load_results(csv_path)
    output_dir = Path("figures")
    ensure_output_dir(output_dir)

    plot_position_error_cdf(df, output_dir)
    plot_error_summary(df, output_dir)
    plot_error_by_position(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_corr_vs_error(df, output_dir)
    plot_corr_histogram(df, output_dir)
    plot_solver_outcomes(df, output_dir)

    print(f"Plots written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
