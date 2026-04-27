"""
Run the backend's analysis pipeline on a single LOPEX93 leaf spectrum.

Demonstrates reflectance, 1st/2nd derivatives, and channel ratios on real
laboratory-grade reference data, validating that the backend produces
physically sensible outputs beyond the synthetic test fixtures.

Run from project root:
    python validate_with_lopex.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis as an
import preprocessing as pp

CSV_PATH    = Path("lopex93.csv")
ROW_INDEX   = 0          # which leaf to use (0 = first reflectance sample)
VISIBLE_LO  = 380
VISIBLE_HI  = 750

OUT_DIR = Path("./lopex_validation")
OUT_DIR.mkdir(exist_ok=True)


def find_wavelength_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        try:
            int(float(str(c).strip()))
            cols.append(c)
        except (ValueError, TypeError):
            pass
    return cols


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Cannot find {CSV_PATH} next to this script.")

    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Reflectance rows only (LOPEX93 mixes reflectance + transmittance)
    df = df[df["Measurement_type"].str.lower() == "reflectance"].reset_index(drop=True)
    print(f"  reflectance rows: {len(df)}")

    wl_cols = find_wavelength_columns(df)
    wl_all  = np.array([int(float(str(c).strip())) for c in wl_cols])

    # Restrict to visible range to match the smartphone spectrometer's coverage
    vis_mask = (wl_all >= VISIBLE_LO) & (wl_all <= VISIBLE_HI)
    wl_vis   = wl_all[vis_mask].astype(float)
    vis_cols = [c for c, keep in zip(wl_cols, vis_mask) if keep]

    # Pull one leaf
    row = df.iloc[ROW_INDEX]
    R   = pd.to_numeric(row[vis_cols], errors="coerce").to_numpy(dtype=float)

    species = row.get("Latin Name", "unknown")
    chl_ab  = row.get("Chlorophyll_a+b (µg/cm²)", float("nan"))
    print(f"\nLeaf {ROW_INDEX}: {species}")
    print(f"  measured chlorophyll a+b: {chl_ab:.2f} µg/cm²")
    print(f"  visible-range points: {len(R)}  ({wl_vis[0]:.0f}–{wl_vis[-1]:.0f} nm)")

    # --- 1. Reflectance + derivatives via the backend's own functions ---
    d1 = pp.sgd(R, order=1)
    d2 = pp.sgd(R, order=2)

    # --- 2. Synthesise per-channel signals from the broadband reflectance ---
    # Map RGB channels to phone-camera centres: B≈460, G≈540, R≈630 nm.
    # We're not splitting the LOPEX93 reflectance; we're sampling it at three
    # representative wavelengths and treating those as the channel "intensities"
    # so we can drive analysis.channel_ratios in a way that reflects what the
    # backend would compute from a phone image of the same leaf.
    def at(wl_target):
        idx = int(np.argmin(np.abs(wl_vis - wl_target)))
        return float(R[idx])

    R_ch = at(630)   # red channel proxy
    G_ch = at(540)   # green channel proxy
    B_ch = at(460)   # blue channel proxy

    EPS = 1e-9
    ratios = {
        "R_mean":   R_ch,
        "G_mean":   G_ch,
        "B_mean":   B_ch,
        "R_over_G": R_ch / max(G_ch, EPS),
        "R_over_B": R_ch / max(B_ch, EPS),
        "G_over_B": G_ch / max(B_ch, EPS),
        "G_over_R": G_ch / max(R_ch, EPS),
    }

    print("\nChannel reflectances (sampled at camera-band centres):")
    print(f"  R(630 nm) = {R_ch:.4f}")
    print(f"  G(540 nm) = {G_ch:.4f}")
    print(f"  B(460 nm) = {B_ch:.4f}")
    print("\nChannel ratios:")
    for k, v in ratios.items():
        if not k.endswith("_mean"):
            print(f"  {k:10s} = {v:.4f}")

    # --- 3. Plot everything for the report ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    ax.plot(wl_vis, R, "b-")
    ax.set_title(f"Reflectance — {species}")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Reflectance")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(wl_vis, d1, "g-")
    ax.axhline(0, c="k", lw=0.5, alpha=0.4)
    ax.set_title("1st derivative (Savitzky-Golay)")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("dR/dλ")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(wl_vis, d2, "r-")
    ax.axhline(0, c="k", lw=0.5, alpha=0.4)
    ax.set_title("2nd derivative (Savitzky-Golay)")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("d²R/dλ²")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    keys = ["R_over_G", "R_over_B", "G_over_B", "G_over_R"]
    ax.bar(keys, [ratios[k] for k in keys], color=["#c0392b","#8e44ad","#2980b9","#27ae60"])
    ax.set_title("Channel ratios")
    ax.set_ylabel("Ratio value")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Backend validation on LOPEX93 leaf {ROW_INDEX} "
                 f"(chlorophyll a+b = {chl_ab:.1f} µg/cm²)",
                 fontsize=12)
    fig.tight_layout()

    out_path = OUT_DIR / f"lopex_leaf_{ROW_INDEX}_validation.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()