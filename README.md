# Hydrographic Heading Correction Case Study

**Video walkthrough**

- [Watch on GitHub (inline player)](https://github.com/thinhrick0101/hydrographic_survey_data_correct/blob/main/reports/video.mp4)
- [Direct raw MP4 link](https://github.com/thinhrick0101/hydrographic_survey_data_correct/raw/main/reports/video.mp4)

This repo documents how I reverse-engineered the Spaarnwoude trial CSVs (no headers), diagnosed the heading sign error, and rebuilt the cable track using a corrected heading. It combines data intuition, geometric reasoning, and visual comparisons of the “before” and “after” cable tracks.

## 1. Data intuition – identifying the columns

The raw CSVs have no headers, so the first step was to infer what each column represents by looking at ranges, units, and correlations across files.

**VCOG (cart pose)**
- `epoch_s`: large, monotonically increasing integer → epoch timestamp.
- `time_of_day`: repeating HHMMSS-like number → time of day.
- `cart_east_m`, `cart_north_m`, `cart_z_m`: large, meter-scale values → ENU cart position.
- `heading_deg`: smooth 0–360° variable that tracks turns → cart heading.
- `aux_1`, `aux_2`: low-variance auxiliary fields.
- `unused`: column with no obvious structure.
- `cable_lock`: trailing 0/1 flag that matches cable status in CCOG.

**CCOG (exported cable track)**
- `epoch_s`, `time_of_day`: same as VCOG.
- `cable_east_m`, `cable_north_m`, `cable_z_m`: large ENU-scale coordinates → exported cable world position.
- `offset_x_m`, `offset_y_m`, `offset_z_m`: small, smooth, meter-scale values; when rotated by heading they land on the cable position → local cable offsets in the cart frame.
- `dir_east`, `dir_north`, `dir_up`: near-unit components (sum of squares ≈ 1) → cable direction vector in ENU.
- `cable_lock`: 0/1 flag matching VCOG.

These mappings are encoded in `analysis.py` as `VCOG_COLS` and `CCOG_COLS`.

## 2. Mathematical / geometric reasoning

### 2.1 Diagnosing the heading

To test whether the heading in the CSV was wrong, I compared:

- **Track heading** from motion: compute `dx`, `dy` between consecutive positions, `dt` from timestamps, and use `track_heading = atan2(dx, dy)` (in degrees).
- **Sensor heading** from the CSV: `heading_deg`.

I formed the residual:

```text
residual = (sensor_heading − track_heading) wrapped to [−180°, 180°]
```

The residuals are centered near zero for moving segments, which shows the logged heading is internally consistent. The issue is not the heading column itself, but how that heading was used to rotate the local cable offsets into world coordinates.

### 2.2 Rebuilding cable position from local offsets

The cable tip is defined in the cart’s local frame by an offset `(x, y, z)`:

- `x`: forward along the cart’s heading
- `y`: lateral (port/starboard)
- `z`: vertical

Given a cart position `(E_cart, N_cart)` and heading `ψ` (in radians), the standard ENU rotation is:

```text
E_offset = x·sin(ψ) + y·cos(ψ)
N_offset = x·cos(ψ) − y·sin(ψ)

E_cable = E_cart + E_offset
N_cable = N_cart + N_offset
```

This is implemented in `rotate_offsets` and used by `rebuild_cable` in `analysis.py`.

### 2.3 Heading correction

An internal note reported a −90° vs +90° mix-up, which effectively flips the orientation by 180°. To correct this, I:

- Treat the CSV heading as “truth”.
- Apply a fixed heading shift before rotation:

```text
heading_correct = heading_csv + heading_fix_deg
# with heading_fix_deg = +180°
ψ = radians(heading_correct)
```

The cable track is then rebuilt using `ψ` in the rotation above. There is also a “reflect” mode that mirrors the published cable across the cart path, mainly to visualize how the wrong heading puts the cable on the wrong side.

## 3. Visualizing “before” vs “after”

`analysis.ipynb` calls into `analysis.py` to generate figures in `reports/figures/`. Key plots:

### 3.1 Plan-view overlays

Files:
- `reports/figures/offset_line_overlays.png` – offset experiments with planned 2 m, 4 m, 6 m separations.
- `reports/figures/toc_overlays.png` – TOC experiments.

Legend:
- Black: cart path (VCOG).
- Orange: published cable (CCOG, before).
- Green: corrected cable (after heading fix).

In the offset runs, the published cable is often on the wrong side of the cart and too close to the path. The corrected cable runs parallel to the cart, on the expected side, with a separation that visually matches the planned 2/4/6 m offsets.

### 3.2 Cross-track distance

Files:
- `reports/figures/cross_track_summary.png` – mean |cross-track| vs planned offset.
- `reports/figures/exp3_cross_track_timeseries.png` – cross-track over time for a 2 m offset run.

Cross-track is the sideways distance from the cart’s path to the cable, measured perpendicular to the heading. The summary plot compares planned offsets (0, 2, 4, 6 m) to the published and corrected mean cross-track. After applying the +180° heading fix, the offset experiments move from “near zero / wrong side” to ≈2 m, ≈3.6 m, and ≈5.0 m, much closer to the planned 2/4/6 m.

### 3.3 Heading residuals

File:
- `reports/figures/heading_vs_track_residuals.png` – histograms of `sensor_heading − track_heading` per experiment.

These histograms are centered near 0°, confirming that the heading values in the CSV are consistent with the cart’s motion and that the correction belongs in the reconstruction math, not in the raw heading.

### 3.4 TOC 1 and TOC 2 improvements

For the TOC runs, the heading fix mainly tightens the cable relative to the cart. In both cases the analysis uses only samples where `cable_lock == 1` (the default `lock_filter=True` in `analysis.py`):

- **TOC 1 (Experiment 1)**: mean horizontal cart–cable distance improves from about 0.22 m (published) to 0.15 m (corrected), and mean cross-track shifts from roughly −0.07 m to −0.01 m. The cable moves from being offset sideways by ~7 cm to almost centered on the cart’s path. This partially recovers a run that was globally warped in the original export.
- **TOC 2 (Experiment 2)**: mean horizontal separation improves from about 0.12 m to 0.07 m, and the cross-track bias shrinks from ~10 cm on one side to ~5 cm on the other, much closer to zero. This turns an already decent export into a clean, well-aligned top-of-cable track.

## 4. What else can I do?

Next things I have done:

- **Uncertainty / error modeling**: attach confidence intervals to cross-track and horizontal distances; propagate heading and position uncertainty through the rotation.
- **Automated QC and reporting**: turn the metrics table into an HTML/PDF report and flag runs where errors exceed thresholds.
- **Compare alternative corrections**: experiment with small heading biases or per-experiment tuning of `heading_fix_deg` and compare cross-track error.
- **Integration with future field trials**: wrap the pipeline into a CLI/CI job so new surveys automatically generate corrected tracks, metrics, and plots when raw files are dropped into `data/raw/`.

## 5. How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # or source .venv/bin/activate on mac/linux
pip install -r requirements.txt
python analysis.py              # prints metrics, regenerates figures, writes corrected CSVs
# Optional knobs inside analysis.py:
#   lock_filter=True (default) keeps only rows where cable_lock == 1
#   smooth_window=N (e.g., 5) applies a short rolling mean to reduce jitter
```

Running `analysis.py` with the default settings also writes heading-corrected CCOG-style files to `data/processed/`, for example:

- `Exp_1_CCoG_TOC_corrected.csv`
- `Exp_2_CCoG_TOC_EW_corrected.csv`
- `Exp_3_CCoG_2m_OL_Ncable_WE_corrected.csv`
- `Exp_4_CCoG_4m_OL_Ncable_EW_corrected.csv`
- `Exp_5_CCoG_6m_OL_Ncable_WE_corrected.csv`

### Notebook

- `analysis.ipynb` is a lightweight entry point that reuses `analysis.py` and displays the saved figures.
