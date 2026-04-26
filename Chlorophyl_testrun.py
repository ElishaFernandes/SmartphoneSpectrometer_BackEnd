"""
Calibrate the chlorophyll preset against the LOPEX93 dataset.

What this script does
---------------------
1. Downloads the LOPEX93 reflectance and metadata spreadsheets from ECOSIS.
2. For every leaf sample, computes the same G/R proxy that the backend uses:
       proxy = mean(reflectance, 530-580 nm) / mean(reflectance, 640-700 nm)
3. Pairs each proxy value with the laboratory-measured Chlorophyll_a+b
   concentration (µg/cm²).
4. Fits a linear regression chlorophyll = m * proxy + b and reports R².
5. Derives data-driven thresholds for the 'low / moderate / high' classes
   from the 33rd and 67th percentiles of the measured chlorophyll values,
   then maps those concentration levels back to proxy values.
6. Saves a calibration plot and a JSON file with the new thresholds, ready
   to drop into analysis.py.

Run from the project root:
    pip install pandas openpyxl requests matplotlib scipy
    python calibrate_chlorophyll.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import linregress

# ---- LOPEX93 resource URLs (from the ECOSIS package metadata) ----
REFLECTANCE_URL = (
    "http://data.ecosis.org/dataset/13aef0ce-dd6f-4b35-91d9-28932e506c41/"
    "resource/b2eb2d47-e790-4d12-bf0e-78402f708796/download/"
    "lopex1993reflectance.xlsx"
)
METADATA_URL = (
    "http://data.ecosis.org/dataset/13aef0ce-dd6f-4b35-91d9-28932e506c41/"
    "resource/79a258a7-7013-4d0c-b5b7-36671821f227/download/"
    "lopex1993metadata.xlsx"
)

# ---- Bands matching the backend's chlorophyll proxy ----
GREEN_BAND = (530, 580)   # nm — chlorophyll reflects here
RED_BAND   = (640, 700)   # nm — chlorophyll absorbs here

OUT_DIR = Path("./calibration_results")
OUT_DIR.mkdir(exist_ok=True)


def fetch_xlsx(url: str) -> pd.ExcelFile:
    print(f"  downloading {url.split('/')[-1]} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.ExcelFile(io.BytesIO(r.content))


def band_mean(spectrum: pd.Series, lo: int, hi: int) -> float:
    """Mean reflectance in [lo, hi] nm. LOPEX93 columns are integer wavelengths."""
    cols = [c for c in spectrum.index if str(c).isdigit() and lo <= int(c) <= hi]
    return float(spectrum[cols].mean())


def main() -> None:
    print("Fetching LOPEX93 ...")
    refl_xl = fetch_xlsx(REFLECTANCE_URL)
    meta_xl = fetch_xlsx(METADATA_URL)

    refl = refl_xl.parse(refl_xl.sheet_names[0])
    meta = meta_xl.parse(meta_xl.sheet_names[0])
    print(f"  reflectance rows: {len(refl)},  metadata rows: {len(meta)}")

    # The reflectance sheet has one row per spectrum; the first column is the
    # sample identifier and the remaining columns are wavelengths in nm.
    sample_col_refl = refl.columns[0]
    refl = refl.set_index(sample_col_refl)

    # Compute G/R proxy per sample
    g = refl.apply(lambda row: band_mean(row, *GREEN_BAND), axis=1)
    r = refl.apply(lambda row: band_mean(row, *RED_BAND), axis=1)
    proxy = (g / r).rename("proxy_G_over_R")

    # Find the chlorophyll a+b column in the metadata sheet
    chl_col = next(c for c in meta.columns if "chlorophyll" in str(c).lower()
                                           and "a+b" in str(c).lower())
    sample_col_meta = meta.columns[0]
    meta = meta.set_index(sample_col_meta)
    chl = pd.to_numeric(meta[chl_col], errors="coerce").rename("chlorophyll_ab")

    # Inner-join on sample id; drop missing values
    df = pd.concat([proxy, chl], axis=1, join="inner").dropna()
    print(f"  paired samples with chlorophyll measurements: {len(df)}")

    # ---- Regression ----
    res = linregress(df["proxy_G_over_R"], df["chlorophyll_ab"])
    print(f"\nLinear fit: chlorophyll = {res.slope:.4f} * proxy + {res.intercept:.4f}")
    print(f"R² = {res.rvalue**2:.4f}    p = {res.pvalue:.2e}    n = {len(df)}")

    # ---- Data-driven thresholds ----
    chl_low_max  = df["chlorophyll_ab"].quantile(0.33)
    chl_mod_max  = df["chlorophyll_ab"].quantile(0.67)
    proxy_low_max = (chl_low_max - res.intercept) / res.slope
    proxy_mod_max = (chl_mod_max - res.intercept) / res.slope
    print("\nRecommended thresholds for analysis.CHLOROPHYLL_THRESHOLDS:")
    print(f"  low_max      = {proxy_low_max:.4f}   (≤ {chl_low_max:.1f} µg/cm²)")
    print(f"  moderate_max = {proxy_mod_max:.4f}   (≤ {chl_mod_max:.1f} µg/cm²)")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["proxy_G_over_R"], df["chlorophyll_ab"], alpha=0.6, s=20)
    xs = np.linspace(df["proxy_G_over_R"].min(), df["proxy_G_over_R"].max(), 100)
    ax.plot(xs, res.slope * xs + res.intercept, "r-",
            label=f"y = {res.slope:.3f}x + {res.intercept:.3f}  (R² = {res.rvalue**2:.3f})")
    ax.axvline(proxy_low_max, ls="--", c="grey", alpha=0.7, label="low/mod boundary")
    ax.axvline(proxy_mod_max, ls="--", c="grey", alpha=0.7, label="mod/high boundary")
    ax.set_xlabel("G/R proxy (mean R 530–580 nm / mean R 640–700 nm)")
    ax.set_ylabel("Chlorophyll a+b (µg/cm²)")
    ax.set_title(f"LOPEX93 chlorophyll calibration (n = {len(df)})")
    ax.legend()
    fig.tight_layout()
    plot_path = OUT_DIR / "chlorophyll_calibration.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")

    # ---- Save calibration JSON ----
    out = {
        "dataset": "LOPEX93",
        "n_samples": int(len(df)),
        "green_band_nm": list(GREEN_BAND),
        "red_band_nm": list(RED_BAND),
        "regression": {
            "slope": float(res.slope),
            "intercept": float(res.intercept),
            "r_squared": float(res.rvalue ** 2),
            "p_value": float(res.pvalue),
        },
        "recommended_thresholds": {
            "low_max": float(proxy_low_max),
            "moderate_max": float(proxy_mod_max),
        },
        "interpretation": {
            "low":      f"chlorophyll a+b ≤ {chl_low_max:.1f} µg/cm²",
            "moderate": f"{chl_low_max:.1f} < chlorophyll a+b ≤ {chl_mod_max:.1f} µg/cm²",
            "high":     f"chlorophyll a+b > {chl_mod_max:.1f} µg/cm²",
        },
    }
    json_path = OUT_DIR / "chlorophyll_calibration.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"Calibration JSON saved to {json_path}")


if __name__ == "__main__":
    main()