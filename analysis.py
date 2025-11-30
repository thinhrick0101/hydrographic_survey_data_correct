from __future__ import annotations

"""
Hydrographic survey heading correction and quick-look analytics.

The raw files come without headers; this module attaches names, applies the
heading correction described in the Spaarnwoude field note, and produces
figures plus a small metrics table for each experiment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# File layout
DATA_DIR = Path("data/raw")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
PROCESSED_DIR = Path("data/processed")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Column guesses from inspecting the files
VCOG_COLS = [
    "epoch_s",
    "time_of_day",
    "cart_east_m",
    "cart_north_m",
    "cart_z_m",
    "heading_deg",
    "aux_1",
    "aux_2",
    "unused",
    "cable_lock",
]

CCOG_COLS = [
    "epoch_s",
    "time_of_day",
    "cable_east_m",
    "cable_north_m",
    "cable_z_m",
    "offset_x_m",
    "offset_y_m",
    "offset_z_m",
    "dir_east",
    "dir_north",
    "dir_up",
    "cable_lock",
]


@dataclass
class Experiment:
    name: str
    vcog_path: Path
    ccog_path: Path
    expected_offset_m: float


EXPERIMENTS: Dict[str, Experiment] = {
    "1_TOC_NS": Experiment(
        "Experiment 1 (TOC)",
        DATA_DIR / "Exp_1_VCoG_TOC.csv",
        DATA_DIR / "Exp_1_CCoG_TOC.csv",
        expected_offset_m=0.0,
    ),
    "2_TOC_EW": Experiment(
        "Experiment 2 (TOC)",
        DATA_DIR / "Exp_2_VCoG_TOC_EW.csv",
        DATA_DIR / "Exp_2_CCoG_TOC_EW.csv",
        expected_offset_m=0.0,
    ),
    "3_2m": Experiment(
        "Experiment 3 (2 m offset)",
        DATA_DIR / "Exp_3_VCoG_2m_OL_Ncable_WE.csv",
        DATA_DIR / "Exp_3_CCoG_2m_OL_Ncable_WE.csv",
        expected_offset_m=2.0,
    ),
    "4_4m": Experiment(
        "Experiment 4 (4 m offset)",
        DATA_DIR / "Exp_4_VCoG_4m_OL_Ncable_EW.csv",
        DATA_DIR / "Exp_4_CCoG_4m_OL_Ncable_EW.csv",
        expected_offset_m=4.0,
    ),
    "5_6m": Experiment(
        "Experiment 5 (6 m offset)",
        DATA_DIR / "Exp_5_VCoG_6m_OL_Ncable_WE.csv",
        DATA_DIR / "Exp_5_CCoG_6m_OL_Ncable_WE.csv",
        expected_offset_m=6.0,
    ),
}


def load_vcog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=VCOG_COLS)
    df["timestamp"] = pd.to_datetime(df["epoch_s"], unit="s", utc=True)
    return df


def load_ccog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=CCOG_COLS)
    df["timestamp"] = pd.to_datetime(df["epoch_s"], unit="s", utc=True)
    return df


def heading_track_residual(cart: pd.DataFrame, min_speed_mps: float = 0.2) -> pd.Series:
    """
    Difference between sensor heading and along-track heading, filtered by speed.
    Positive = sensor heading is clockwise relative to the path heading.
    """
    dx = cart["cart_east_m"].diff()
    dy = cart["cart_north_m"].diff()
    dt = cart["timestamp"].diff().dt.total_seconds()
    speed = np.hypot(dx, dy) / dt
    track_heading = np.rad2deg(np.arctan2(dx, dy))
    sensor_heading = cart["heading_deg"]
    residual = (sensor_heading - track_heading + 540.0) % 360.0 - 180.0
    mask = (speed > min_speed_mps) & residual.notna()
    return residual[mask]


def apply_qc(
    vcog: pd.DataFrame,
    ccog: pd.DataFrame,
    lock_filter: bool = True,
    smooth_window: int = 1,
    heading_smooth_window: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optional QC step: drop low-confidence rows (cable_lock != 1) and smooth
    positions/offsets with a short rolling mean to reduce jitter.
    """
    vcog = vcog.copy()
    ccog = ccog.copy()

    if lock_filter:
        mask = ccog["cable_lock"] == 1
        vcog = vcog.loc[mask].reset_index(drop=True)
        ccog = ccog.loc[mask].reset_index(drop=True)

    if smooth_window and smooth_window > 1:
        offset_cols = ["offset_x_m", "offset_y_m", "offset_z_m"]
        pose_cols = ["cart_east_m", "cart_north_m", "cart_z_m", "heading_deg"]
        ccog[offset_cols] = (
            ccog[offset_cols].rolling(window=smooth_window, center=True, min_periods=1).mean()
        )
        vcog[pose_cols] = (
            vcog[pose_cols].rolling(window=smooth_window, center=True, min_periods=1).mean()
        )

    if heading_smooth_window and heading_smooth_window > 1:
        # Smooth heading on the circle via mean sine/cosine
        ang = np.deg2rad(vcog["heading_deg"])
        sin_roll = pd.Series(np.sin(ang)).rolling(
            window=heading_smooth_window, center=True, min_periods=1
        ).mean()
        cos_roll = pd.Series(np.cos(ang)).rolling(
            window=heading_smooth_window, center=True, min_periods=1
        ).mean()
        smoothed = np.rad2deg(np.arctan2(sin_roll, cos_roll))
        vcog["heading_deg"] = smoothed

    return vcog, ccog


def rotate_offsets(
    offset_x: np.ndarray, offset_y: np.ndarray, heading_deg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate a local (x forward, y starboard) offset into East/North using heading.
    """
    theta = np.deg2rad(heading_deg)
    east = offset_x * np.sin(theta) + offset_y * np.cos(theta)
    north = offset_x * np.cos(theta) - offset_y * np.sin(theta)
    return east, north


def cross_track(cart_e: pd.Series, cart_n: pd.Series, heading_deg: pd.Series, target_e: np.ndarray, target_n: np.ndarray) -> np.ndarray:
    """
    Signed cross-track distance of the target relative to cart heading.
    Positive = target on port (left) side when looking along heading.
    """
    theta = np.deg2rad(heading_deg)
    left_e = -np.sin(theta)
    left_n = np.cos(theta)
    de = target_e - cart_e
    dn = target_n - cart_n
    return de * left_e + dn * left_n


def rebuild_cable(
    vcog: pd.DataFrame, ccog: pd.DataFrame, heading_shift_deg: float
) -> pd.DataFrame:
    """
    Reconstruct cable position from cart pose, local offsets, and a heading correction.
    heading_shift_deg is added to the heading in the VCOG file:
    - Use -90 if the heading is a raw baseline heading that needs aligning to the cart frame.
    - Use +180 to undo the documented -90 vs +90 mix-up (flip the heading used for CCOG).
    """
    east_off, north_off = rotate_offsets(
        ccog["offset_x_m"].to_numpy(),
        ccog["offset_y_m"].to_numpy(),
        vcog["heading_deg"].to_numpy() + heading_shift_deg,
    )
    rebuilt = pd.DataFrame(
        {
            "cable_east_m": vcog["cart_east_m"].to_numpy() + east_off,
            "cable_north_m": vcog["cart_north_m"].to_numpy() + north_off,
            "cable_z_m": vcog["cart_z_m"].to_numpy() + ccog["offset_z_m"].to_numpy(),
        },
        index=vcog.index,
    )
    return rebuilt


def reflect_cable(vcog: pd.DataFrame, ccog: pd.DataFrame) -> pd.DataFrame:
    """
    Side-flip only: reflect the exported cable solution across the cart track.
    Keeps the cart-cable separation magnitude but swaps the side.
    """
    reflected = pd.DataFrame(
        {
            "cable_east_m": 2 * vcog["cart_east_m"].to_numpy() - ccog["cable_east_m"].to_numpy(),
            "cable_north_m": 2 * vcog["cart_north_m"].to_numpy() - ccog["cable_north_m"].to_numpy(),
            "cable_z_m": ccog["cable_z_m"].to_numpy(),
        },
        index=vcog.index,
    )
    return reflected


def run_figures(
    summary: pd.DataFrame,
    carts: Dict[str, pd.DataFrame],
    corrected_positions: Dict[str, pd.DataFrame],
    published_positions: Dict[str, pd.DataFrame],
    correction_label: str,
) -> None:
    # Top-of-cable overlay
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key in zip(axes, ["1_TOC_NS", "2_TOC_EW"]):
        cart = carts[key]
        published = published_positions[key]
        corrected = corrected_positions[key]
        ax.plot(cart["cart_east_m"], cart["cart_north_m"], label="Cart (VCOG)", lw=2)
        ax.plot(published["east"], published["north"], label="Exported CCOG", alpha=0.6)
        ax.plot(corrected["east"], corrected["north"], label=correction_label, alpha=0.8)
        ax.set_title(EXPERIMENTS[key].name)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.axis("equal")
        ax.legend()
    fig.suptitle("Top-of-cable lines: exported vs heading-corrected cable track", fontsize=12)
    fig.savefig(FIG_DIR / "top_of_cable_overlay.png", dpi=200)
    plt.close(fig)

    # Top-of-cable overlay with shared legend (Exp 1/2)
    toc_keys = ["1_TOC_NS", "2_TOC_EW"]
    fig, axes = plt.subplots(1, len(toc_keys), figsize=(11, 4.2), constrained_layout=True)
    for ax, key in zip(axes, toc_keys):
        cart = carts[key]
        published = published_positions[key]
        corrected = corrected_positions[key]
        ax.plot(cart["cart_east_m"], cart["cart_north_m"], label="Cart (VCOG)", lw=2, color="k")
        ax.plot(published["east"], published["north"], label="Exported CCOG", alpha=0.65, color="C1")
        ax.plot(corrected["east"], corrected["north"], label=correction_label, alpha=0.8, color="C2")
        ax.set_title(f'{EXPERIMENTS[key].name} (plan 0 m)')
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.axis("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Top-of-cable: cart vs cable before/after heading fix", fontsize=12)
    fig.savefig(FIG_DIR / "toc_overlays.png", dpi=200)
    plt.close(fig)

    # Offset-line overlays (cart vs published vs corrected)
    offset_keys = ["3_2m", "4_4m", "5_6m"]
    fig, axes = plt.subplots(1, len(offset_keys), figsize=(15, 4.2), constrained_layout=True)
    for ax, key in zip(axes, offset_keys):
        cart = carts[key]
        published = published_positions[key]
        corrected = corrected_positions[key]
        ax.plot(cart["cart_east_m"], cart["cart_north_m"], label="Cart (VCOG)", lw=2, color="k")
        ax.plot(published["east"], published["north"], label="Exported CCOG", alpha=0.65, color="C1")
        ax.plot(corrected["east"], corrected["north"], label=correction_label, alpha=0.8, color="C2")
        ax.set_title(f'{EXPERIMENTS[key].name} (plan {EXPERIMENTS[key].expected_offset_m:.0f} m)')
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.axis("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Offset lines: cart vs cable before/after heading fix", fontsize=12)
    fig.savefig(FIG_DIR / "offset_line_overlays.png", dpi=200)
    plt.close(fig)

    # Cross-track magnitude vs plan
    fig, ax = plt.subplots(figsize=(7, 4))
    order = list(EXPERIMENTS.keys())
    expected = [EXPERIMENTS[k].expected_offset_m for k in order]
    before = (
        summary[summary["label"] == "published"]
        .set_index("experiment_key")["mean_cross_track_m"]
        .reindex(order)
    )
    after = (
        summary[summary["label"] == "corrected"]
        .set_index("experiment_key")["mean_cross_track_m"]
        .reindex(order)
    )
    ax.bar(np.arange(len(order)) - 0.15, before.abs(), width=0.3, label="Published |cross-track|")
    ax.bar(np.arange(len(order)) + 0.15, after.abs(), width=0.3, label="Corrected |cross-track|")
    ax.plot(np.arange(len(order)), expected, "k--", label="Planned offset")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("Mean |cross-track| (m)")
    ax.set_title("Cable lateral offset vs plan")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_track_summary.png", dpi=200)
    plt.close(fig)

    # Example time-series for first offset line
    key = "3_2m"
    cart = carts[key]
    published = published_positions[key]
    corrected = corrected_positions[key]
    xt_pub = cross_track(
        cart["cart_east_m"], cart["cart_north_m"], cart["heading_deg"], published["east"], published["north"]
    )
    xt_cor = cross_track(
        cart["cart_east_m"], cart["cart_north_m"], cart["heading_deg"], corrected["east"], corrected["north"]
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(cart["timestamp"], xt_pub, label="Published")
    ax.plot(cart["timestamp"], xt_cor, label=correction_label)
    ax.axhline(EXPERIMENTS[key].expected_offset_m, color="k", ls="--", label="Planned offset")
    ax.set_ylabel("Cross-track (m)")
    ax.set_xlabel("Time")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "exp3_cross_track_timeseries.png", dpi=200)
    plt.close(fig)

    # Heading sign error illustration on an offset line (shows side swap)
    key = "3_2m"
    cart = carts[key]
    published = published_positions[key]
    corrected = corrected_positions[key]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(cart["cart_east_m"], cart["cart_north_m"], label="Cart (VCOG)", lw=2, color="k")
    ax.plot(published["east"], published["north"], label="Exported (+90°)", alpha=0.7, color="C1")
    ax.plot(corrected["east"], corrected["north"], label=correction_label, alpha=0.8, color="C2")
    ax.set_title("Heading sign error flips cable to wrong side (Exp 3, 2 m)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "heading_orientation_issue.png", dpi=200)
    plt.close(fig)


def run_heading_diagnostics(carts: Dict[str, pd.DataFrame], min_speed_mps: float = 0.2) -> None:
    """
    Histogram of heading residuals (sensor - track) per experiment to show internal consistency.
    """
    fig, axes = plt.subplots(2, 3, figsize=(11, 5), constrained_layout=True)
    axes = axes.ravel()
    keys = list(EXPERIMENTS.keys())
    for idx, key in enumerate(keys):
        ax = axes[idx]
        resid = heading_track_residual(carts[key], min_speed_mps=min_speed_mps)
        if resid.empty:
            ax.text(0.5, 0.5, "No data after speed filter", ha="center", va="center")
            continue
        ax.hist(resid, bins=np.arange(-20, 21, 1), color="C0", alpha=0.8)
        ax.axvline(resid.median(), color="k", ls="--", lw=1, label=f"Median {resid.median():.1f}°")
        ax.set_title(EXPERIMENTS[key].name, fontsize=9)
        ax.set_xlabel("Heading residual (deg)")
        ax.set_ylabel("Count")
        ax.legend()
    # Hide unused subplot slot if any
    for j in range(len(keys), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Sensor heading vs along-track heading (speed > {min_speed_mps:.1f} m/s)", fontsize=12)
    fig.savefig(FIG_DIR / "heading_vs_track_residuals.png", dpi=200)
    plt.close(fig)


def main(
    heading_fix_deg: float = 180.0,
    generate_figures: bool = True,
    lock_filter: bool = True,
    smooth_window: int = 1,
    heading_smooth_window: int = 1,
    correction_mode: str = "heading_shift",
    generate_heading_diag: bool = True,
    heading_residual_speed_mps: float = 0.2,
    write_corrected: bool = False,
    xt_error_threshold_m: float = 1.0,
    write_report: bool = True,
) -> pd.DataFrame:
    """
    heading_fix_deg is added to the heading column from the CSV:
    - Default +180: matches the current files, undoing the -90 vs +90 mix-up (flips heading used for CCOG).
    - Set to -90: only if the CSV heading is raw baseline and needs aligning to the cart frame.
    correction_mode: "heading_shift" applies heading_fix_deg; "reflect" mirrors CCOG across cart.
    write_corrected: if True and correction_mode == "heading_shift", write corrected CCOG-style CSVs
        to DATA_DIR/../processed with updated cable_east/north/z columns.
    xt_error_threshold_m: absolute cross-track error above this is flagged in the report.
    write_report: if True, write CSV + HTML metrics summary into the reports folder.
    """
    all_metrics = []
    carts: Dict[str, pd.DataFrame] = {}
    ccogs: Dict[str, pd.DataFrame] = {}
    published_positions: Dict[str, pd.DataFrame] = {}
    corrected_positions: Dict[str, pd.DataFrame] = {}
    correction_label = (
        f"Corrected ({heading_fix_deg:+.0f} deg)"
        if correction_mode == "heading_shift"
        else "Reflected across cart"
    )

    for key, exp in EXPERIMENTS.items():
        vcog = load_vcog(exp.vcog_path)
        ccog = load_ccog(exp.ccog_path)
        vcog, ccog = apply_qc(
            vcog,
            ccog,
            lock_filter=lock_filter,
            smooth_window=smooth_window,
            heading_smooth_window=heading_smooth_window,
        )
        if correction_mode == "heading_shift":
            rebuilt = rebuild_cable(vcog, ccog, heading_fix_deg)
        elif correction_mode == "reflect":
            rebuilt = reflect_cable(vcog, ccog)
        else:
            raise ValueError(f"Unknown correction_mode '{correction_mode}'")
        published = ccog[["cable_east_m", "cable_north_m", "cable_z_m"]].rename(
            columns={"cable_east_m": "east", "cable_north_m": "north", "cable_z_m": "z"}
        )
        corrected = rebuilt.rename(columns={"cable_east_m": "east", "cable_north_m": "north", "cable_z_m": "z"})

        carts[key] = vcog
        ccogs[key] = ccog
        published_positions[key] = published
        corrected_positions[key] = corrected

        for label, cable in [("published", published), ("corrected", corrected)]:
            horiz = np.hypot(
                cable["east"] - vcog["cart_east_m"],
                cable["north"] - vcog["cart_north_m"],
            )
            xt = cross_track(
                vcog["cart_east_m"],
                vcog["cart_north_m"],
                vcog["heading_deg"],
                cable["east"],
                cable["north"],
            )
            all_metrics.append(
                {
                    "experiment_key": key,
                    "experiment": exp.name,
                    "n_samples": len(horiz),
                    "label": label,
                    "expected_offset_m": exp.expected_offset_m,
                    "mean_horizontal_m": horiz.mean(),
                    "std_horizontal_m": horiz.std(),
                    "mean_cross_track_m": xt.mean(),
                    "std_cross_track_m": xt.std(),
                    "abs_cross_track_error_m": abs(abs(xt.mean()) - exp.expected_offset_m),
                    "heading_fix_deg": heading_fix_deg,
                    "correction_mode": correction_mode,
                }
            )

    summary = pd.DataFrame(all_metrics)

    # Simple 95% confidence intervals (empirical) for the mean distances:
    # CI ≈ 1.96 * std / sqrt(n)
    n = summary["n_samples"].replace(0, np.nan)
    summary["mean_horizontal_ci95_m"] = 1.96 * summary["std_horizontal_m"] / np.sqrt(n)
    summary["mean_cross_track_ci95_m"] = 1.96 * summary["std_cross_track_m"] / np.sqrt(n)

    # Simple QC flag based on cross-track error
    summary["cross_track_ok"] = summary["abs_cross_track_error_m"] <= xt_error_threshold_m

    # Persist metrics for downstream use / CI
    metrics_csv = REPORT_DIR / "metrics_summary.csv"
    summary.to_csv(metrics_csv, index=False)

    if write_report:
        html_path = REPORT_DIR / "metrics_summary.html"
        with html_path.open("w", encoding="utf-8") as f:
            f.write("<html><head><title>Heading correction metrics</title></head><body>\n")
            f.write("<h1>Heading correction metrics</h1>\n")
            f.write("<p>Rows with <code>cross_track_ok == False</code> have an absolute "
                    f"cross-track error above {xt_error_threshold_m:.2f} m.</p>\n")
            f.write(summary.to_html(index=False, float_format="{:.3f}".format))
            flagged = summary[~summary["cross_track_ok"]]
            if not flagged.empty:
                f.write("<h2>Flagged experiments</h2>\n")
                f.write(flagged.to_html(index=False, float_format="{:.3f}".format))
            f.write("</body></html>\n")

    if write_corrected and correction_mode == "heading_shift":
        for key, exp in EXPERIMENTS.items():
            reb = corrected_positions[key]
            ccog = ccogs[key].copy()
            ccog[["cable_east_m", "cable_north_m", "cable_z_m"]] = reb[
                ["east", "north", "z"]
            ].to_numpy()
            out_path = PROCESSED_DIR / f"{exp.ccog_path.stem}_corrected.csv"
            ccog.to_csv(out_path, header=False, index=False)

    if generate_figures:
        run_figures(summary, carts, corrected_positions, published_positions, correction_label)
        if generate_heading_diag:
            run_heading_diagnostics(carts, min_speed_mps=heading_residual_speed_mps)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hydrographic heading correction quick-look pipeline")
    parser.add_argument("--heading-fix-deg", type=float, default=180.0, help="Heading correction in degrees")
    parser.add_argument(
        "--correction-mode",
        choices=["heading_shift", "reflect"],
        default="heading_shift",
        help="Apply a heading shift or just reflect cable across cart",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Do not generate matplotlib figures",
    )
    parser.add_argument(
        "--no-heading-diag",
        action="store_true",
        help="Do not generate heading residual diagnostics",
    )
    parser.add_argument(
        "--no-write-corrected",
        action="store_true",
        help="Do not write corrected CCOG-style CSVs",
    )
    parser.add_argument(
        "--xt-error-threshold-m",
        type=float,
        default=1.0,
        help="Absolute cross-track error threshold used for QC flagging",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write CSV/HTML summary reports",
    )

    args = parser.parse_args()

    summary_df = main(
        heading_fix_deg=args.heading_fix_deg,
        generate_figures=not args.no_figures,
        correction_mode=args.correction_mode,
        generate_heading_diag=not args.no_heading_diag,
        write_corrected=not args.no_write_corrected,
        xt_error_threshold_m=args.xt_error_threshold_m,
        write_report=not args.no_report,
    )
    pd.set_option("display.float_format", "{:.3f}".format)
    print(
        summary_df[summary_df["label"] == "corrected"][
            [
                "experiment",
                "expected_offset_m",
                "mean_cross_track_m",
                "std_cross_track_m",
                "mean_horizontal_m",
                "abs_cross_track_error_m",
                "cross_track_ok",
            ]
        ]
    )
