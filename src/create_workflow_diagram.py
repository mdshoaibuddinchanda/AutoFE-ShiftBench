"""Create the AutoFE-ShiftBench workflow diagram for paper submission.

The output is deliberately generated with Matplotlib primitives so that the
PDF remains vector-based and the PNG can be exported at a controlled 300 DPI.
No existing result is removed or overwritten; the two new assets use distinct
filenames in ``results``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY = "#19324A"
BLUE = "#2F6F9F"
TEAL = "#2A9D8F"
ORANGE = "#E58F3A"
GREEN = "#5A9367"
PALE_BLUE = "#EAF2F8"
PALE_TEAL = "#E8F5F2"
PALE_ORANGE = "#FDF1E4"
PALE_GREEN = "#EEF6EE"
TEXT = "#17212B"


def _box(ax, x: float, y: float, w: float, h: float, title: str, body: str, *,
         edge: str, face: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    # The workflow is placed at manuscript width.  Use print-safe typography
    # and a compact vertical rhythm so the labels remain legible after the
    # vector PDF is reduced from the source canvas to a single text column.
    if h <= 0.17:
        title_y, body_y = y + h - 0.020, y + 0.035
        title_fs, body_fs, line_spacing = 13.0, 11.0, 1.08
    elif h <= 0.19:
        title_y, body_y = y + h - 0.018, y + 0.055
        title_fs, body_fs, line_spacing = 13.2, 11.5, 1.10
    elif h <= 0.22:
        title_y, body_y = y + h - 0.028, y + h / 2 - 0.006
        title_fs, body_fs, line_spacing = 15.5, 13.5, 1.15
    else:
        title_y, body_y = y + h - 0.040, y + h / 2 - 0.005
        title_fs, body_fs, line_spacing = 18, 15, 1.25
    ax.text(
        x + w / 2, title_y, title,
        transform=ax.transAxes, ha="center", va="top", color=edge,
        fontsize=title_fs, fontweight="bold", family="DejaVu Sans", zorder=3,
    )
    ax.text(
        x + w / 2, body_y, body,
        transform=ax.transAxes, ha="center", va="center", color=TEXT,
        fontsize=body_fs, linespacing=line_spacing, family="DejaVu Sans", zorder=3,
    )


def _arrow(ax, x1: float, x2: float, y: float, color: str = NAVY) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y), (x2, y),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=24,
            linewidth=2.2,
            color=color,
            shrinkA=3,
            shrinkB=3,
            zorder=1,
        )
    )


def _connector(ax, start: tuple[float, float], end: tuple[float, float], *,
               color: str = NAVY, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end,
            transform=ax.transAxes,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=2.0,
            color=color,
            shrinkA=4,
            shrinkB=4,
            zorder=1,
        )
    )


def create_workflow_diagram(out_dir: str | Path = "results", dpi: int = 300) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    # This is an internal method diagram rather than a project-management
    # checklist: it exposes the data path, representation branches, fitting,
    # evaluation, and aggregation used by every completed task.
    fig, ax = plt.subplots(figsize=(10, 9), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(
        0.5, 0.955,
        "AutoFE-ShiftBench: internal evaluation protocol",
        ha="center", va="top", color=NAVY, fontsize=21, fontweight="bold",
    )
    fig.text(
        0.5, 0.915,
        "Fold-local fitting; predictor-wide stress partitions are transductive",
        ha="center", va="top", color="#52616B", fontsize=12.5,
    )

    # Stage 1: data and shift control.
    ax.text(0.03, 0.865, "DATA AND SHIFT CONTROL", transform=ax.transAxes,
            ha="left", va="bottom", color=NAVY, fontsize=12.5, fontweight="bold")
    top_y, top_h, top_w = 0.66, 0.18, 0.205
    top_x = [0.03, 0.275, 0.52, 0.765]
    top = [
        ("INPUT\nDATA", "OpenML tables\nfeatures + target\nconfigured tasks", BLUE, PALE_BLUE),
        ("REPEATED\nSPLITS", "5 seeds × 5 folds\nstratified / stress splits\npaired task key", TEAL, PALE_TEAL),
        ("TRAINING\nCORRUPTION", "noise · missing · relabel\nremoval; test unchanged\n14 conditions incl. splits", ORANGE, PALE_ORANGE),
        ("FOLD-LOCAL\nPREP", "impute · encode\nscale on train only\ncache by task key", GREEN, PALE_GREEN),
    ]
    for x, (title, body, edge, face) in zip(top_x, top):
        _box(ax, x, top_y, top_w, top_h, title, body, edge=edge, face=face)
    for i in range(3):
        _arrow(ax, top_x[i] + top_w + 0.008, top_x[i + 1] - 0.008, top_y + top_h / 2)

    # Stage 2: the prepared fold branches into the capped raw controls and
    # the explicit AutoFE candidate/selection space.
    ax.text(0.03, 0.605, "REPRESENTATION AND MODEL FITTING", transform=ax.transAxes,
            ha="left", va="bottom", color=NAVY, fontsize=12.5, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5}, zorder=5)
    branch_y, branch_h, branch_w = 0.42, 0.17, 0.31
    _box(ax, 0.08, branch_y, branch_w, branch_h, "RAW CONTROLS",
         "Raw · Raw_Full · Raw_Capped\nvariance / MI controls\nno synthesis operators",
         edge=BLUE, face=PALE_BLUE)
    _box(ax, 0.61, branch_y, branch_w, branch_h, "AUTOFE SEARCH",
         "DFS depth-one arithmetic\n+/−/×/÷; selection\n100-column final cap",
         edge=GREEN, face=PALE_GREEN)
    _connector(ax, (top_x[-1] + top_w / 2, top_y - 0.008),
               (0.08 + branch_w / 2, branch_y + branch_h + 0.008), rad=0.18)
    _connector(ax, (top_x[-1] + top_w / 2, top_y - 0.008),
               (0.61 + branch_w / 2, branch_y + branch_h + 0.008), rad=-0.08)

    model_x, model_y, model_w, model_h = 0.355, 0.245, 0.29, 0.16
    _box(ax, model_x, model_y, model_w, model_h, "MODEL FITTING",
         "10 classifiers\nCPU + GPU families\none fit per task",
         edge=NAVY, face="#F7F9FB")
    _connector(ax, (0.08 + branch_w / 2, branch_y - 0.008),
               (model_x + model_w / 2, model_y + model_h + 0.008), rad=-0.12)
    _connector(ax, (0.61 + branch_w / 2, branch_y - 0.008),
               (model_x + model_w / 2, model_y + model_h + 0.008), rad=0.12)

    # Stage 3: every fit produces held-out evidence, a task ledger, and then
    # data-set-level summaries used by the manuscript.
    ax.text(0.03, 0.225, "EVALUATION AND EVIDENCE", transform=ax.transAxes,
            ha="left", va="bottom", color=NAVY, fontsize=12.5, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.5}, zorder=5)
    bottom_y, bottom_h, bottom_w = 0.035, 0.16, 0.205
    bottom_x = [0.03, 0.275, 0.52, 0.765]
    bottom = [
        ("HELD-OUT\nEVALUATION", "ROC-AUC · F1\ntrain–test gap\nclean test"),
        ("TASK\nLEDGER", "runtime + RSS\nfeature counts\nJSONL + SQLite"),
        ("DATASET-LEVEL\nANALYSIS", "balanced means\nrobustness + ablation\npaired contrasts"),
        ("PAPER\nEVIDENCE", "tables · figures\nFSVA diagnostics\nLaTeX manuscript"),
    ]
    for x, (title, body) in zip(bottom_x, bottom):
        _box(ax, x, bottom_y, bottom_w, bottom_h, title, body, edge=NAVY, face="#F7F9FB")
    _connector(ax, (model_x + model_w / 2, model_y - 0.008),
               (bottom_x[0] + bottom_w / 2, bottom_y + bottom_h + 0.008), rad=0.20)
    for i in range(3):
        _arrow(ax, bottom_x[i] + bottom_w + 0.008, bottom_x[i + 1] - 0.008,
               bottom_y + bottom_h / 2, color="#52616B")

    png_path = out_dir / "AutoFE_ShiftBench_Workflow.png"
    pdf_path = out_dir / "AutoFE_ShiftBench_Workflow.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png, pdf = create_workflow_diagram()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
