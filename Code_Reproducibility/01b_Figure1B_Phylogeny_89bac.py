# /// script
# requires-python = ">=3.10"
# dependencies = ["biopython", "matplotlib", "numpy", "pandas"]
# ///
"""
01b_Figure1B_Phylogeny_89bac.py
===============================
Circular phylogenomic tree of 89 non-redundant bacterial MAGs (Figure 1B).
All tips are displayed as an equal-branch-length circular cladogram with four concentric tracks
(Host plant, CheckM2 completeness, CheckM2 contamination, and Genome size) and clades colored by phylum.

Output:
    Figures/Panels/Fig1B_Phylogeny_89MAGs.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from Bio import Phylo

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Arial"

PKG_DIR = Path(__file__).resolve().parent.parent
RES  = PKG_DIR / "Source_Data" / "results"
FIG  = PKG_DIR / "Figures" / "Panels"
FIG2 = PKG_DIR / "Figures" / "Panels"

TREE_FILE  = RES / "Phylogenetic_Tree_89bac_MAGs.nwk"
META_FILE  = RES / "Metadata_89bac_MAGs.tsv"
PAL_FILE   = RES / "Phylum_Color_Palette.tsv"
ABUND_FILE = RES / "MAG_Phylum_relative_abundance_89bac_12samples.tsv"

OPEN_ANGLE_DEG = 10.0        # degrees – gap at root (350° circle)
TARGET_RADIUS  = 5.5         # tree radius in plot units
HOST_COL = {"Citrus": "#E53935", "Coffee": "#1E88E5"}

# ─────────────────────────────────────────────────────────────────
# Tree layout helpers
# ─────────────────────────────────────────────────────────────────

def count_terminals(clade) -> int:
    if clade.is_terminal():
        return 1
    return sum(count_terminals(c) for c in clade.clades)


def subtree_depth(clade) -> int:
    """Max # edges from clade to any tip (0 for terminals)."""
    if clade.is_terminal():
        return 0
    return 1 + max(subtree_depth(c) for c in clade.clades)


def assign_angles(clade, a_start: float, a_end: float) -> None:
    """Recursively assign _angle (midpoint of subtree span) to every node."""
    n = count_terminals(clade)
    if clade.is_terminal():
        clade._angle = (a_start + a_end) / 2
        return
    a = a_start
    for child in clade.clades:
        nc = count_terminals(child)
        a_child_end = a + (nc / n) * (a_end - a_start)
        assign_angles(child, a, a_child_end)
        a = a_child_end
    # Internal node: midpoint of its full subtree range
    clade._angle = (a_start + a_end) / 2


def assign_radii(clade, max_depth: int) -> None:
    """Dendrogram layout: all tips at TARGET_RADIUS, internals inward."""
    d = subtree_depth(clade)
    clade._radius = TARGET_RADIUS * (max_depth - d) / max_depth
    for child in clade.clades:
        assign_radii(child, max_depth)


def get_all_nodes(clade):
    yield clade
    for child in clade.clades:
        yield from get_all_nodes(child)


def compute_layout(tree) -> None:
    open_rad = np.radians(OPEN_ANGLE_DEG)
    # Gap at the right (0° axis, 3 o'clock): start at half_gap, end at 2π - half_gap
    half_gap  = open_rad / 2
    a_start   = half_gap
    a_end     = 2 * np.pi - half_gap

    root = tree.root
    assign_angles(root, a_start, a_end)
    md = subtree_depth(root)
    assign_radii(root, md)


# ─────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────

def draw_arc(ax, r: float, a1: float, a2: float, color: str, lw: float) -> None:
    """Draw an arc at radius r from angle a1 to a2."""
    if abs(a2 - a1) < 1e-6:
        return
    n = max(int(abs(a2 - a1) * 120) + 4, 6)
    angs = np.linspace(a1, a2, n)
    ax.plot(r * np.cos(angs), r * np.sin(angs),
            color=color, lw=lw, solid_capstyle="round", zorder=2)


def draw_radial(ax, r1: float, r2: float, angle: float, color: str, lw: float) -> None:
    ax.plot([r1 * np.cos(angle), r2 * np.cos(angle)],
            [r1 * np.sin(angle), r2 * np.sin(angle)],
            color=color, lw=lw, solid_capstyle="round", zorder=2)


def draw_clade(ax, clade, parent_r: float, color: str = "#222222", lw: float = 0.9) -> None:
    """Recursively draw circular phylogram branches.

    For each internal node:
      - One arc at node's radius spanning from leftmost to rightmost child angle
      - One radial line per child (from node's radius to child's radius)
    """
    if not clade.clades:
        return
    child_angles = [child._angle for child in clade.clades]
    # Horizontal arc connecting all children at this node's radius
    draw_arc(ax, clade._radius, min(child_angles), max(child_angles), color, lw)
    for child in clade.clades:
        # Vertical radial line from this node outward to child
        draw_radial(ax, clade._radius, child._radius, child._angle, color, lw)
        draw_clade(ax, child, clade._radius, color, lw)


def draw_ring_sector(ax, ri: float, ro: float,
                     a1: float, a2: float, color: str) -> None:
    """Filled annular sector from angle a1 to a2, inner ri outer ro."""
    n = max(int(abs(a2 - a1) * 120) + 4, 6)
    angs = np.linspace(a1, a2, n)
    xs = list(ri * np.cos(angs)) + list(ro * np.cos(angs[::-1]))
    ys = list(ri * np.sin(angs)) + list(ro * np.sin(angs[::-1]))
    ax.fill(xs, ys, color=color, lw=0, zorder=3)


def draw_ring(ax, ri: float, ro: float,
              tip_names: list[str], tip_angles: dict[str, float],
              colors_map: dict[str, str]) -> None:
    """Draw a full ring with one sector per tip, colored by colors_map."""
    n = len(tip_names)
    if n == 0:
        return
    sorted_tips = sorted(tip_names, key=lambda t: tip_angles[t])
    all_ang = [tip_angles[t] for t in sorted_tips]
    step = (all_ang[-1] - all_ang[0]) / max(n - 1, 1) if n > 1 else 0.01
    for i, t in enumerate(sorted_tips):
        a_mid = tip_angles[t]
        a1 = a_mid - step / 2
        a2 = a_mid + step / 2
        col = colors_map.get(t, "#AAAAAA")
        draw_ring_sector(ax, ri, ro, a1, a2, col)
    # Ring outlines
    open_half = np.radians(OPEN_ANGLE_DEG) / 2
    # use a continuous arc covering the tip range
    arc_a1 = all_ang[0] - step / 2
    arc_a2 = all_ang[-1] + step / 2
    n_arc = max(int(abs(arc_a2 - arc_a1) * 200) + 10, 20)
    arc_angs = np.linspace(arc_a1, arc_a2, n_arc)
    for r in [ri, ro]:
        ax.plot(r * np.cos(arc_angs), r * np.sin(arc_angs),
                color="#555555", lw=0.4, zorder=4)


def draw_bar_sector(ax, ri: float, h: float, max_h: float,
                    a1: float, a2: float, color: str) -> None:
    ro = ri + (h / max_h) * max_h
    draw_ring_sector(ax, ri, ro, a1, a2, color)


def draw_clade_arc(ax, ri: float, ro: float, angles: list[float],
                   color: str, step: float, label: str, letter: str) -> None:
    """Filled arc for a phylum clade with letter label outside."""
    if not angles:
        return
    a1 = min(angles) - step / 2
    a2 = max(angles) + step / 2
    n = max(int(abs(a2 - a1) * 200) + 10, 10)
    angs = np.linspace(a1, a2, n)
    xs = list(ri * np.cos(angs)) + list(ro * np.cos(angs[::-1]))
    ys = list(ri * np.sin(angs)) + list(ro * np.sin(angs[::-1]))
    ax.fill(xs, ys, color=matplotlib.colors.to_rgba(color, 0.40), lw=0, zorder=1)

    # Letter label outside the arc
    mid_a = (a1 + a2) / 2
    label_r = ro + 0.55
    deg = np.degrees(mid_a)
    rot = deg if (-90 <= deg <= 90) else deg + 180
    ax.text(label_r * np.cos(mid_a), label_r * np.sin(mid_a), letter,
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="black", rotation=rot, clip_on=False, zorder=6)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load tree ────────────────────────────────────────────────
    tree = Phylo.read(str(TREE_FILE), "newick")
    root = tree.root
    # Ensure all branch lengths are set (uniform = 1 if absent)
    for clade in tree.find_clades():
        if clade.branch_length is None:
            clade.branch_length = 1.0

    # ── Load metadata ────────────────────────────────────────────
    meta = pd.read_csv(META_FILE, sep="\t")
    meta["mag"] = meta["mag"].astype(str).str.strip()
    meta["phylum"]      = meta["phylum"].fillna("Unknown").replace("", "Unknown")
    meta["host"]        = meta["host"].fillna("Coffee").str.capitalize()
    meta["completeness"]= meta["completeness"].fillna(75.0)
    meta["contamination"]= meta["contamination"].fillna(0.0)
    meta["length"]      = meta["length"].fillna(0.0)

    # Map mag → metadata
    meta_map = meta.set_index("mag").to_dict("index")

    # ── Load phylum color palette ────────────────────────────────
    pal_df  = pd.read_csv(PAL_FILE, sep="\t")
    pal_map = dict(zip(pal_df["phylum"], pal_df["color"]))

    # ── Phylum order: by mean abundance (same as stacked bar) ────
    tip_names = [c.name for c in tree.get_terminals()]
    phyla_present = sorted({meta_map.get(t, {}).get("phylum", "Unknown") for t in tip_names}
                           - {"Unknown"})
    try:
        abund = pd.read_csv(ABUND_FILE, sep="\t", index_col=0)
        ordered = abund.mean().sort_values(ascending=False).index.tolist()
        phyla = [p for p in ordered if p in phyla_present] + \
                sorted(p for p in phyla_present if p not in ordered)
    except Exception:
        phyla = sorted(phyla_present)

    # Fallback colors for phyla not in palette
    import itertools
    fallback = itertools.cycle([
        "#A6CEE3","#1F78B4","#B2DF8A","#33A02C","#FB9A99","#E31A1C",
        "#FDBF6F","#FF7F00","#CAB2D6","#6A3D9A","#FFFF99","#B15928",
    ])
    clade_pal: dict[str, str] = {}
    for p in phyla:
        clade_pal[p] = pal_map.get(p, next(fallback))

    phylum_letters = {p: chr(ord("a") + i) for i, p in enumerate(phyla)}

    # ── Compute layout ───────────────────────────────────────────
    compute_layout(tree)

    # Collect tip angles
    tip_angles: dict[str, float] = {}
    for c in tree.get_terminals():
        tip_angles[c.name] = c._angle

    sorted_tips = sorted(tip_names, key=lambda t: tip_angles[t])
    all_angs = [tip_angles[t] for t in sorted_tips]
    step = (all_angs[-1] - all_angs[0]) / max(len(all_angs) - 1, 1)

    R_max_tree = TARGET_RADIUS  # all tips at this radius

    # ── Ring parameters (mirrors R script) ───────────────────────
    r_clade_in  = 0.05
    r_clade_out = R_max_tree + 0.4
    ring_w   = 0.65
    ring_gap = 0.12
    r_host_in  = r_clade_out + 1.2
    r_host_out = r_host_in  + ring_w
    r_comp_in  = r_host_out + ring_gap
    r_comp_out = r_comp_in  + ring_w
    r_cont_in  = r_comp_out + ring_gap
    r_cont_out = r_cont_in  + ring_w
    r_bar_in   = r_cont_out + ring_gap * 1.5
    h_bar_max  = 2.20

    # Color arrays
    comp_vals  = np.array([meta_map.get(t, {}).get("completeness", 75.0)  for t in sorted_tips])
    cont_vals  = np.array([meta_map.get(t, {}).get("contamination", 0.0)  for t in sorted_tips])
    len_vals   = np.array([meta_map.get(t, {}).get("length", 0.0)          for t in sorted_tips])
    host_vals  = [meta_map.get(t, {}).get("host", "Coffee")               for t in sorted_tips]

    # Completeness colormap: white→blue; Contamination colormap: white→green
    from matplotlib.colors import LinearSegmentedColormap
    comp_cmap  = LinearSegmentedColormap.from_list("comp", ["#EFF3FF", "#084594"])
    cont_cmap  = LinearSegmentedColormap.from_list("cont", ["#EDF8E9", "#238B45"])
    comp_colors = {t: matplotlib.colors.to_hex(comp_cmap(comp_vals[i] / 100))
                   for i, t in enumerate(sorted_tips)}
    max_cont = max(cont_vals) if max(cont_vals) > 0 else 1.0
    cont_colors = {t: matplotlib.colors.to_hex(cont_cmap(cont_vals[i] / max_cont))
                   for i, t in enumerate(sorted_tips)}
    host_colors = {t: HOST_COL.get(host_vals[i], "#1E88E5")
                   for i, t in enumerate(sorted_tips)}

    max_len = max(len_vals) if max(len_vals) > 0 else 1.0

    # ── Figure ───────────────────────────────────────────────────
    R_lim = 18
    fig, ax = plt.subplots(figsize=(32, 32))
    ax.set_xlim(-R_lim, R_lim)
    ax.set_ylim(-R_lim, R_lim)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Clade arcs (phylum coloring, innermost ring) ─────────────
    for p in phyla:
        tips_in_phylum = [t for t in tip_names if meta_map.get(t, {}).get("phylum") == p]
        angs = [tip_angles[t] for t in tips_in_phylum]
        if not angs:
            continue
        draw_clade_arc(ax, r_clade_in, r_clade_out, angs,
                       clade_pal[p], step, p, phylum_letters[p])

    # ── Tree branches ────────────────────────────────────────────
    draw_clade(ax, root, root._radius, color="#222222", lw=0.9)

    # ── Outer rings ──────────────────────────────────────────────
    draw_ring(ax, r_host_in,  r_host_out,  sorted_tips, tip_angles, host_colors)
    draw_ring(ax, r_comp_in,  r_comp_out,  sorted_tips, tip_angles, comp_colors)
    draw_ring(ax, r_cont_in,  r_cont_out,  sorted_tips, tip_angles, cont_colors)

    # ── Genome size bars ─────────────────────────────────────────
    max_h = h_bar_max
    for i, t in enumerate(sorted_tips):
        h = len_vals[i]
        if h <= 0:
            continue
        a_mid = tip_angles[t]
        a1 = a_mid - step / 2
        a2 = a_mid + step / 2
        ro_bar = r_bar_in + (h / max_len) * max_h
        draw_ring_sector(ax, r_bar_in, ro_bar, a1, a2, "#AAAAAA")

    # ── Legends – compact, all within axes bounds (-18..18) ─────
    FS_TITLE = 8    # pt – section titles
    FS_ITEM  = 7    # pt – legend items
    SQ_W = 0.65     # patch width  (data units)
    SQ_H = 0.85     # patch height (data units)
    ROW_H = 1.35    # row height   (data units)
    SEC_GAP = 0.8   # gap between sections

    # ── Right-side legends (Host, Completeness, Contamination, Genome Size) ──
    lx = 13.2
    y_cur = R_lim * 0.90

    def _patch(x, y, color):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y - SQ_H / 2), SQ_W, SQ_H,
            boxstyle="square,pad=0", fc=color, ec="none",
            clip_on=False, zorder=10))

    # Host
    ax.text(lx, y_cur, "Host Species", fontsize=FS_TITLE, fontweight="bold",
            va="top", ha="left", clip_on=False)
    y_cur -= ROW_H * 0.55
    for h, c in HOST_COL.items():
        y_cur -= ROW_H
        _patch(lx, y_cur, c)
        ax.text(lx + SQ_W + 0.3, y_cur, h, fontsize=FS_ITEM, va="center",
                ha="left", clip_on=False)
    y_cur -= SEC_GAP

    # Completeness
    ax.text(lx, y_cur, "Completeness (%)", fontsize=FS_TITLE, fontweight="bold",
            va="top", ha="left", clip_on=False)
    y_cur -= ROW_H * 0.55
    for lbl, val in [("100%", 1.0), ("75%", 0.75), ("50%", 0.50)]:
        y_cur -= ROW_H
        _patch(lx, y_cur, matplotlib.colors.to_hex(comp_cmap(val)))
        ax.text(lx + SQ_W + 0.3, y_cur, lbl, fontsize=FS_ITEM, va="center",
                ha="left", clip_on=False)
    y_cur -= SEC_GAP

    # Contamination
    ax.text(lx, y_cur, "Contamination (%)", fontsize=FS_TITLE, fontweight="bold",
            va="top", ha="left", clip_on=False)
    y_cur -= ROW_H * 0.55
    for lbl, val in [(f"{max_cont:.0f}%", 1.0), (f"{max_cont/2:.0f}%", 0.5), ("0%", 0.0)]:
        y_cur -= ROW_H
        _patch(lx, y_cur, matplotlib.colors.to_hex(cont_cmap(val)))
        ax.text(lx + SQ_W + 0.3, y_cur, lbl, fontsize=FS_ITEM, va="center",
                ha="left", clip_on=False)
    y_cur -= SEC_GAP

    # Genome size
    ax.text(lx, y_cur, "Genome Size", fontsize=FS_TITLE, fontweight="bold",
            va="top", ha="left", clip_on=False)
    min_len = min(v for v in len_vals if v > 0) if any(v > 0 for v in len_vals) else 0
    y_cur -= ROW_H * 0.55
    for lbl, lw_v in [(f"Max: {max_len/1e6:.2f} Mbp", 5), (f"Min: {min_len/1e6:.2f} Mbp", 2)]:
        y_cur -= ROW_H
        ax.plot([lx, lx + SQ_W], [y_cur, y_cur],
                color="#AAAAAA", lw=lw_v, solid_capstyle="butt", clip_on=False)
        ax.text(lx + SQ_W + 0.3, y_cur, lbl, fontsize=FS_ITEM, va="center",
                ha="left", clip_on=False)

    # ── Scale bar in gap region (bottom, 270°) ────────────────────
    sb_len = 0.55
    sy = -(r_bar_in + h_bar_max + 1.5)
    ax.plot([-sb_len / 2, sb_len / 2], [sy, sy], lw=2.5, color="#333333", clip_on=False)
    ax.plot([-sb_len / 2, -sb_len / 2], [sy - 0.07, sy + 0.07], lw=1.5, color="#333333",
            clip_on=False)
    ax.plot([ sb_len / 2,  sb_len / 2], [sy - 0.07, sy + 0.07], lw=1.5, color="#333333",
            clip_on=False)
    ax.text(0, sy - 0.20, "0.1", ha="center", va="top", fontsize=7,
            color="#333333", clip_on=False)
    ax.text(0, sy - 0.85, "Bacteria", ha="center", va="top",
            fontsize=8, fontweight="bold", fontstyle="normal",
            color="#111111", clip_on=False)

    # ── Add 1, 2, 3, 4 in the gap at 3 o'clock (angle = 0) ───────
    ax.text(r_host_in + ring_w/2, 0, "1", ha="center", va="center", fontsize=13, fontweight="bold", color="black")
    ax.text(r_comp_in + ring_w/2, 0, "2", ha="center", va="center", fontsize=13, fontweight="bold", color="black")
    ax.text(r_cont_in + ring_w/2, 0, "3", ha="center", va="center", fontsize=13, fontweight="bold", color="black")
    ax.text(r_bar_in + 0.5, 0, "4", ha="center", va="center", fontsize=13, fontweight="bold", color="black")

    # ── Phylum legend – 2-column layout on the left ───────────────
    FS_PHY  = 6.5
    SQ_WP   = 0.50
    SQ_HP   = 0.65
    ROW_HP  = 1.05
    COL_W   = 6.0    # x offset between columns
    lx_left = -17.5  # within -18..18
    n_col1  = (len(phyla) + 1) // 2
    y_phy   = R_lim * 0.82

    ax.text(lx_left, y_phy + 0.85, "Phylum", fontsize=FS_TITLE, fontweight="bold",
            va="bottom", ha="left", clip_on=False)
    for i, p in enumerate(phyla):
        col_i = i // n_col1
        row_i = i % n_col1
        xp = lx_left + col_i * COL_W
        yp = y_phy - row_i * ROW_HP
        ax.add_patch(mpatches.FancyBboxPatch(
            (xp, yp - SQ_HP / 2), SQ_WP, SQ_HP,
            boxstyle="square,pad=0", fc=clade_pal[p], ec="none",
            clip_on=False, zorder=10))
        ax.text(xp + SQ_WP + 0.25, yp, f"{phylum_letters[p]}: {p}",
                fontsize=FS_PHY, va="center", ha="left",
                color="#111111", clip_on=False)

    # ── Save ─────────────────────────────────────────────────────
    FIG.mkdir(parents=True, exist_ok=True)
    FIG2.mkdir(parents=True, exist_ok=True)
    out1 = FIG / "Fig1B_Phylogeny_89MAGs.png"
    for out in [out1]:
        fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out}")
    plt.close(fig)
    print(f"Tips: {len(tip_names)}  |  Phyla: {len(phyla)}  |  open_angle: {OPEN_ANGLE_DEG}°")


if __name__ == "__main__":
    main()
