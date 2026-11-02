"""Shared typography and PNG/SVG/PDF export for the paper notebooks."""
from pathlib import Path
from textwrap import fill
import sys

import matplotlib as mpl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = PROJECT_ROOT / "result"
FONT = {
    "base": 13, "title": 15, "header": 15.5, "subtitle": 13.5,
    "label": 13, "tick": 12.5, "legend": 12.5, "annotation": 12.5,
    "note": 12.5, "colorbar_title": 14.5,
}
AUDIT_SCRIPTS = Path("/data/home/wangyongyan/.codex/skills/nature-figure/scripts")
if str(AUDIT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AUDIT_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


def apply_paper_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FONT["base"], "axes.titlesize": FONT["title"],
        "axes.labelsize": FONT["label"], "xtick.labelsize": FONT["tick"],
        "ytick.labelsize": FONT["tick"], "legend.fontsize": FONT["legend"],
        "legend.title_fontsize": FONT["label"], "figure.titlesize": FONT["title"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "savefig.dpi": 300,
    })


def wrap_note(text, width=100):
    return fill(text, width=width)


def figure_qa_stem(stem):
    relative = Path(stem).resolve().relative_to(RESULT_ROOT.resolve())
    target = RESULT_ROOT / "_qa" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_paper_figure(fig, stem, *, exclude_axes=(), check_alignment=True, dpi=300):
    """Save the same figure bounds in all three formats under result/."""
    stem = Path(stem).resolve()
    qa = figure_qa_stem(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    if check_alignment:
        require_matplotlib_panel_alignment(
            fig, exclude_axes=list(exclude_axes),
            json_out=qa.with_suffix(".alignment.json"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        )
    options = {"dpi": dpi, "bbox_inches": "tight", "facecolor": "white"}
    fig.savefig(stem.with_suffix(".png"), **options)
    fig.savefig(stem.with_suffix(".svg"), **options)
    fig.savefig(stem.with_suffix(".pdf"), **options)
