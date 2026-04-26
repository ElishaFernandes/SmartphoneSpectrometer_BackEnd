"""
Saving and exporting.

CSV: wavelength + reflectance + derivatives where present.
PDF: ReportLab document with metadata, ratios table, preset extras, and
matplotlib-rendered curves embedded as PNGs.

Storage paths are configurable via env vars:
    EXPORT_DIR  default ./storage/exports
    GRAPH_DIR   default ./storage/graphs
"""
from __future__ import annotations

import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend - safe for headless servers
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle,
)


EXPORT_DIR = os.environ.get("EXPORT_DIR", "./storage/exports")
GRAPH_DIR = os.environ.get("GRAPH_DIR", "./storage/graphs")
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ---------- CSV ----------

def export_csv(record: dict) -> str:
    """Write a CSV with wavelength, reflectance, and derivatives where present."""
    sid = record["id"]
    path = os.path.join(EXPORT_DIR, f"{sid}.csv")
    wl = record.get("wavelength_data") or []
    R = record.get("reflectance_data") or []
    d1 = record.get("derivative_data")
    d2 = record.get("second_derivative_data")

    headers = ["wavelength_nm", "reflectance"]
    if d1 is not None:
        headers.append("first_derivative")
    if d2 is not None:
        headers.append("second_derivative")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        n = len(wl)
        for i in range(n):
            row = [wl[i], R[i] if i < len(R) else ""]
            if d1 is not None:
                row.append(d1[i] if i < len(d1) else "")
            if d2 is not None:
                row.append(d2[i] if i < len(d2) else "")
            w.writerow(row)
    return path


# ---------- PNG helpers for the PDF ----------

def _save_curve_png(wl, y, title, ylabel, fname) -> str:
    path = os.path.join(GRAPH_DIR, fname)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(wl, y, linewidth=1.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_ratio_bar_png(ratios: dict, fname: str) -> str:
    path = os.path.join(GRAPH_DIR, fname)
    keys = ["R_over_G", "R_over_B", "G_over_B", "G_over_R"]
    vals = [float(ratios.get(k, 0.0)) for k in keys]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(keys, vals)
    ax.set_ylabel("ratio")
    ax.set_title("Channel ratios")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------- PDF ----------

def export_pdf(record: dict) -> str:
    """Build a one-page-ish PDF report for the session."""
    sid = record["id"]
    pdf_path = os.path.join(EXPORT_DIR, f"{sid}.pdf")

    wl = np.asarray(record.get("wavelength_data") or [])
    R = np.asarray(record.get("reflectance_data") or [])
    d1 = record.get("derivative_data")
    d2 = record.get("second_derivative_data")
    ratios = record.get("ratios") or {}
    extras = record.get("extras") or {}

    images = []
    if len(wl) and len(R):
        images.append(_save_curve_png(wl, R, "Reflectance", "R(λ)", f"{sid}_R.png"))
    if d1 is not None:
        images.append(_save_curve_png(wl, np.asarray(d1), "1st derivative", "dR/dλ", f"{sid}_d1.png"))
    if d2 is not None:
        images.append(_save_curve_png(wl, np.asarray(d2), "2nd derivative", "d²R/dλ²", f"{sid}_d2.png"))
    if ratios:
        images.append(_save_ratio_bar_png(ratios, f"{sid}_ratios.png"))

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    story = []
    story.append(Paragraph(f"Spectrometer report — {sid}", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))

    integrated = record.get("integrated_reflectance") or 0.0
    meta = [
        ["Preset", record.get("preset", "")],
        ["Timestamp", record.get("timestamp", "")],
        ["Concentration", str(record.get("concentration"))],
        ["Integrated reflectance", f"{integrated:.4f}"],
    ]
    t = Table(meta, hAlign="LEFT", colWidths=[5 * cm, 10 * cm])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    if ratios:
        story.append(Paragraph("Channel ratios", styles["Heading2"]))
        rs = [[k, f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)]
              for k, v in ratios.items()]
        rt = Table(rs, hAlign="LEFT", colWidths=[5 * cm, 5 * cm])
        rt.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ]))
        story.append(rt)
        story.append(Spacer(1, 0.4 * cm))

    if extras:
        story.append(Paragraph("Preset extras", styles["Heading2"]))
        for k, v in extras.items():
            story.append(Paragraph(
                f"<b>{k}</b>: <font face='Courier' size='8'>{v}</font>",
                styles["BodyText"],
            ))
        story.append(Spacer(1, 0.4 * cm))

    for img_path in images:
        story.append(RLImage(img_path, width=15 * cm, height=8 * cm))
        story.append(Spacer(1, 0.3 * cm))

    doc.build(story)
    return pdf_path