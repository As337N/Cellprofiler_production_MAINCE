"""
qc/output/pdf.py — Lightweight, email-attachable PDF summary.

The self-contained HTML report is great for exploration but large and sometimes
blocked by mail filters. This module renders a compact PDF (a few pages) with
what a reviewer needs at a glance:
  · A per-plate summary table: n_wells, pass rate, illum pass, red-well counts.
  · Each plate's overview thumbnail (already in the payload as base64).
  · The flagged wells per plate with their absolute/adaptive failure tags.

Built from the SAME window.__QC__ payload as the HTML, so the two never diverge.

Engine: matplotlib's PdfPages — pure Python, no native system dependencies
beyond matplotlib itself, which is typically already present.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Page geometry (A4 landscape, inches).
_PAGE_W, _PAGE_H = 11.69, 8.27
_RED_KEYS = [("slope", "Slope"), ("medint", "MedInt"),
             ("pctmax", "PctMax"), ("pctmin", "PctMin")]


def summary_pdf(payload: dict, output_path: Path) -> None:
    """Render the executive-summary PDF from a window.__QC__ payload.

    Parameters
    ----------
    payload : dict
        The same structure produced by qc.payload.builder.build_payload.
    output_path : Path
        Destination .pdf file.
    """
    data = payload.get("data", {})
    red_counts = payload.get("redCounts", {})
    plates = sorted(data.keys())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        _summary_table_page(pdf, plates, data, red_counts)
        for plate in plates:
            _plate_page(pdf, plate, data[plate], red_counts)

    print(f"[pdf] Summary written to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Page 1: per-plate summary table
# ─────────────────────────────────────────────────────────────────────────────
def _summary_table_page(pdf: PdfPages, plates: list, data: dict,
                        red_counts: dict) -> None:
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
    fig.suptitle("Cell Painting QC — Summary", fontsize=16, fontweight="bold",
                 y=0.97)

    header = ["Plate", "Wells", "Pass", "Pass %", "Illum"] + \
             [lbl for _, lbl in _RED_KEYS]
    rows = []
    for p in plates:
        d = data[p]
        rows.append([
            p,
            str(d.get("n_wells", 0)),
            str(d.get("n_pass", 0)),
            f"{d.get('pass_rate', 0):.1f}",
            str(d.get("n_illum", 0)),
            *[str(red_counts.get(key, {}).get(p, 0)) for key, _ in _RED_KEYS],
        ])

    ax = fig.add_axes([0.05, 0.05, 0.90, 0.85])
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "No plates processed.", ha="center", va="center")
        pdf.savefig(fig)
        plt.close(fig)
        return

    table = ax.table(cellText=rows, colLabels=header, loc="upper center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Header styling + pass-rate colour cue.
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a2233")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 3:  # Pass % column
            try:
                pr = float(rows[r - 1][3])
                cell.set_facecolor("#d7f5dd" if pr >= 80 else
                                   "#ffe9c7" if pr >= 50 else "#ffd6d6")
            except (ValueError, IndexError):
                pass

    pdf.savefig(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# One page per plate: overview thumbnail + flagged-well list
# ─────────────────────────────────────────────────────────────────────────────
def _plate_page(pdf: PdfPages, plate: str, d: dict, red_counts: dict) -> None:
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
    fig.suptitle(f"Plate {plate}", fontsize=14, fontweight="bold", y=0.97)

    # Left: overview thumbnail.
    ax_img = fig.add_axes([0.03, 0.05, 0.55, 0.85])
    ax_img.axis("off")
    img = _decode_b64_image(d.get("overview_b64"))
    if img is not None:
        ax_img.imshow(img)
    else:
        ax_img.text(0.5, 0.5, "No overview image", ha="center", va="center")

    # Right: stats block + flagged wells.
    ax_txt = fig.add_axes([0.60, 0.05, 0.37, 0.85])
    ax_txt.axis("off")

    lines = [
        f"Wells: {d.get('n_wells', 0)}",
        f"Pass: {d.get('n_pass', 0)}  ({d.get('pass_rate', 0):.1f}%)",
        f"Illum pass: {d.get('n_illum', 0)}",
        "",
        "Red-well counts:",
    ]
    for key, lbl in _RED_KEYS:
        lines.append(f"   {lbl}: {red_counts.get(key, {}).get(plate, 0)}")

    flags = d.get("well_flags", {})
    lines.append("")
    lines.append(f"Flagged wells ({len(flags)}):")
    if flags:
        for well in sorted(flags)[:25]:
            f = flags[well]
            tags = ", ".join(f.get("abs", []) + f.get("adp", []))
            lines.append(f"   {well}: {tags[:60]}")
        if len(flags) > 25:
            lines.append(f"   … and {len(flags) - 25} more")
    else:
        lines.append("   none")

    ax_txt.text(0.0, 1.0, "\n".join(lines), ha="left", va="top",
                family="monospace", fontsize=9, transform=ax_txt.transAxes)

    pdf.savefig(fig)
    plt.close(fig)


def _decode_b64_image(b64: "str | None"):
    """Decode a base64 (JPEG/PNG) string to an array matplotlib can imshow."""
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        return plt.imread(io.BytesIO(raw), format="jpeg")
    except Exception:
        try:
            return plt.imread(io.BytesIO(raw))
        except Exception:
            return None
