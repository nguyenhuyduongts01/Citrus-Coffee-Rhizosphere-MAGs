"""
01_Figure1_panels_and_composite.py
==================================
Reproduces Figure 1 panels and the 4-panel composite Figure 1:
  - Panel A: Phylum-level relative abundance of 89 non-redundant bacterial MAGs across S1-S10
  - Panel B: Phylogenomic tree of 89 bacterial MAGs (GTDB-Tk bac120, rendered via ggtree/ggtreeExtra in 01b_Figure1B_Phylogeny_89bac.R)
  - Panel C: CheckM2 Completeness vs. Contamination of the 17 HQ MAGs by Host Species and by Plant Health Status (Symptomatic vs. Asymptomatic)
  - Panel D: COG functional category equal-petal distribution across the 17 HQ MAGs

Outputs:
  - Figures/Fig1.png (600 DPI)
  - Figures/Fig1.tif (600 DPI, LZW-compressed RGB TIFF, PACE-compliant)
  - Standalone panel images in Figures/Panels/
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

PKG_DIR = Path(__file__).resolve().parent.parent
RES  = PKG_DIR / "Source_Data" / "results"
OUT_FIG = PKG_DIR / "Figures"
OUT_PNL = OUT_FIG / "Panels"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_PNL.mkdir(parents=True, exist_ok=True)

# Unified Phylum Color Palette (loaded from Phylum_Color_Palette.tsv; consistent across Fig 1, 5, 6, 8, S5, S6)
PHYLUM_COLORS = {
    "Pseudomonadota":       "#E41A1C",
    "Actinomycetota":       "#377EB8",
    "Acidobacteriota":      "#4DAF4A",
    "Gemmatimonadota":      "#F781BF",
    "Bacteroidota":         "#984EA3",
    "Verrucomicrobiota":    "#A65628",
    "Desulfobacterota":     "#666666",
    "Nitrospirota":         "#00CED1",
    "Eremiobacterota":      "#B3DE69",
    "CSP1-3":               "#CAB2D6",
    "Desulfobacterota_B":   "#999999",
    "Planctomycetota":      "#FF7F00",
    "Chloroflexota":        "#FFFF33",
    "Dependentiae":         "#FDBF6F",
    "Methylomirabilota":    "#CCEBC5",
    "Patescibacteria":      "#6A3D9A",
    "Other":                "#BDBDBD",
}
_pal_file = Path(__file__).parent / "Phylum_Color_Palette.tsv"
if _pal_file.exists():
    _pal_df = pd.read_csv(_pal_file, sep="\t")
    PHYLUM_COLORS.update(dict(zip(_pal_df["phylum"], _pal_df["color"])))

PHYLUM_LETTERS = [
    ("a", "Pseudomonadota"),
    ("b", "Actinomycetota"),
    ("c", "Acidobacteriota"),
    ("d", "Gemmatimonadota"),
    ("e", "Bacteroidota"),
    ("f", "Verrucomicrobiota"),
    ("g", "Desulfobacterota"),
    ("h", "Nitrospirota"),
    ("i", "Eremiobacterota"),
    ("j", "CSP1-3"),
    ("k", "Desulfobacterota_B"),
    ("l", "Planctomycetota"),
    ("m", "Chloroflexota"),
    ("n", "Dependentiae"),
    ("o", "Methylomirabilota"),
    ("p", "Patescibacteria"),
]

SAMPLES_10 = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]

HQ17 = [
    "Sg_S1_3", "Sg_S3_1", "Sg_S3_8", "Sg_S4_6", "Sg_S5_2",
    "Sg_S6_2", "Sg_S6_5", "Sg_S7_1", "Sg_S7_10", "Sg_S7_19",
    "Sg_S7_2", "Sg_S7_20", "Sg_S7_28", "Sg_S7_7", "Sg_S7_8",
    "Sg_S7_9", "Sg_S9_9"
]


def save_pace_tiff(png_path: Path, tif_path: Path, max_w: int = 4500, max_h: int = 5250, dpi: int = 600):
    """Convert PNG to PLOS ONE PACE-compliant RGB TIFF with LZW compression and synchronized PNG dimensions."""
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0 or w < 2000:
        target_scale = min(max_w / w, max_h / h)
        new_w = int(round(w * target_scale))
        new_h = int(round(h * target_scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img.save(png_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    img.save(tif_path, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))



def draw_panel_a(ax_bar, ax_leg):
    """Draw Panel A: 10-sample MAG Phylum stacked bar + lettered Phylum legend."""
    mag_abund = pd.read_csv(RES / "MAG_relative_abundance_89bac.tsv", sep="\t")
    mag_abund["mag_id"] = (
        mag_abund["mag_id"].astype(str)
        .str.replace("Sg_", "", regex=False)
        .str.replace(".fa", "", regex=False)
    )
    meta89 = pd.read_csv(RES / "Metadata_89bac_MAGs.tsv", sep="\t")
    meta89["mag"] = (
        meta89["mag"].astype(str)
        .str.replace("Sg_", "", regex=False)
        .str.replace(".fa", "", regex=False)
    )
    mag_phylum = meta89[["mag", "phylum"]].drop_duplicates().set_index("mag")
    mag_abund = mag_abund.merge(mag_phylum, left_on="mag_id", right_index=True, how="left")
    mag_abund["phylum"] = mag_abund["phylum"].fillna("Other")

    mag_abund_10 = mag_abund[mag_abund["sample"].isin(SAMPLES_10)].copy()
    phy_agg = mag_abund_10.groupby(["sample", "phylum"])["relative_abundance"].sum().reset_index()
    ppiv = phy_agg.pivot_table(
        index="sample", columns="phylum", values="relative_abundance", aggfunc="sum", fill_value=0
    ).reindex(SAMPLES_10).fillna(0)
    ppct = ppiv.div(ppiv.sum(axis=1), axis=0).fillna(0) * 100.0

    # Stack order: bottom = most abundant (Pseudomonadota on top as in Figure 1A: reversed stack)
    phylum_order = [p for _, p in PHYLUM_LETTERS if p in ppct.columns]
    stack_order = list(reversed(phylum_order))

    x = np.arange(len(SAMPLES_10))
    bottom = np.zeros(len(SAMPLES_10))
    for phy in stack_order:
        vals = ppct[phy].values
        ax_bar.bar(
            x, vals, bottom=bottom, width=0.68,
            color=PHYLUM_COLORS.get(phy, "#BDBDBD"),
            edgecolor="white", linewidth=0.3
        )
        bottom += vals

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(SAMPLES_10, fontsize=11.5)
    ax_bar.set_ylim(0, 102)
    ax_bar.set_xlim(-0.6, len(SAMPLES_10) - 0.4)
    ax_bar.set_ylabel("Relative Abundance (%)", fontsize=13.5, fontweight="bold")
    ax_bar.set_xlabel("Sample", fontsize=13.5, fontweight="bold")
    ax_bar.tick_params(axis="y", labelsize=11)

    # Lettered Phylum text legend (a: Pseudomonadota ... p: Patescibacteria)
    ax_leg.axis("off")
    ax_leg.text(0.02, 0.96, "Phylum", fontsize=15, fontweight="bold", va="top", ha="left")
    for idx, (letter, phy) in enumerate(PHYLUM_LETTERS):
        y_pos = 0.89 - idx * 0.054
        ax_leg.text(
            0.02, y_pos, f"{letter}:",
            fontsize=12.5, fontweight="bold", va="top", ha="left", color="#111111"
        )
        ax_leg.text(
            0.14, y_pos, phy,
            fontsize=12.5, fontweight="bold", fontstyle="normal", va="top", ha="left", color="#111111"
        )


def draw_panel_c(ax_host, ax_status):
    """Draw Panel C: Completeness vs. Contamination of 17 HQ MAGs (Host & Plant Health Status)."""
    meta = pd.read_csv(RES / "Metadata_17HQ_MAGs.tsv", sep="\t")
    meta["host_clean"] = meta["host"].astype(str).str.strip().str.lower()
    meta["cond_clean"] = meta["condition"].astype(str).str.strip().str.lower()

    # Left: By Host Species
    host_cfg = [
        ("citrus", "Citrus", "#E41A1C"),
        ("coffee", "Coffee", "#183BF0"),
    ]
    for key, label, col in host_cfg:
        sub = meta[meta["host_clean"] == key]
        ax_host.scatter(
            sub["contamination"], sub["completeness"],
            c=col, label=label, s=68, alpha=0.92, edgecolors="none", zorder=3
        )
    ax_host.axhline(90, color="#888888", linestyle="--", linewidth=0.85, alpha=0.7)
    ax_host.axvline(5, color="#888888", linestyle="--", linewidth=0.85, alpha=0.7)
    ax_host.set_xlim(-0.2, 6.0)
    ax_host.set_ylim(88, 102)
    ax_host.set_xlabel("Contamination (%)", fontsize=12.5, fontweight="bold")
    ax_host.set_ylabel("Completeness (%)", fontsize=12.5, fontweight="bold")
    ax_host.set_title("By Host Species", fontsize=12.5, fontweight="bold")
    ax_host.grid(True, color="#EAEAEA", linewidth=0.6, zorder=0)
    ax_host.spines["top"].set_visible(False)
    ax_host.spines["right"].set_visible(False)
    ax_host.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95)

    # Right: By Plant Health Status (Symptomatic vs. Asymptomatic)
    status_cfg = [
        (["disease", "diseased", "symptomatic"], "Symptomatic", "#808080"),
        (["healthy", "asymptomatic"], "Asymptomatic", "#90EE90"),
    ]
    for keys, label, col in status_cfg:
        sub = meta[meta["cond_clean"].isin(keys)]
        ax_status.scatter(
            sub["contamination"], sub["completeness"],
            c=col, label=label, s=68, alpha=0.92, edgecolors="none", zorder=3
        )
    ax_status.axhline(90, color="#888888", linestyle="--", linewidth=0.85, alpha=0.7)
    ax_status.axvline(5, color="#888888", linestyle="--", linewidth=0.85, alpha=0.7)
    ax_status.set_xlim(-0.2, 6.0)
    ax_status.set_ylim(88, 102)
    ax_status.set_xlabel("Contamination (%)", fontsize=12.5, fontweight="bold")
    ax_status.set_ylabel("Completeness (%)", fontsize=12.5, fontweight="bold")
    ax_status.set_title("By Plant Health Status", fontsize=12.5, fontweight="bold")
    ax_status.grid(True, color="#EAEAEA", linewidth=0.6, zorder=0)
    ax_status.spines["top"].set_visible(False)
    ax_status.spines["right"].set_visible(False)
    ax_status.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.95)


def draw_panel_d(ax):
    """Draw Panel D: COG Functional Equal-Petal Plot for 17 HQ MAGs."""
    cog = pd.read_csv(RES / "COG_analysis_22_MAGs.tsv", sep="\t")
    mag_col = "mag_id" if "mag_id" in cog.columns else "mag"
    cog17 = cog[cog[mag_col].isin(HQ17)].copy()

    info_stor = list("JKLABT")
    cell_proc = list("DYVTMNZWUOX")
    metabolism = list("CEFGHIPQ")

    group_color = {
        "Information storage": "#6BAED6",
        "Cellular processes":  "#74C476",
        "Metabolism":          "#FB6A4A",
        "Poorly characterized": "#BDBDBD",
    }
    def get_group(letter):
        if letter in info_stor: return "Information storage"
        if letter in cell_proc: return "Cellular processes"
        if letter in metabolism: return "Metabolism"
        return "Poorly characterized"

    totals = cog17.groupby("cog_letter")["gene_count"].sum().reset_index()
    totals.columns = ["letter", "count"]
    totals = totals[totals["count"] > 0].copy()
    totals["group"] = totals["letter"].apply(get_group)
    totals["pct"] = totals["count"] / totals["count"].sum() * 100.0

    group_order = ["Information storage", "Cellular processes", "Metabolism", "Poorly characterized"]
    totals["g_idx"] = totals["group"].apply(lambda g: group_order.index(g))
    totals = totals.sort_values(["g_idx", "letter"]).reset_index(drop=True)

    ax.set_aspect("equal")
    ax.axis("off")

    N = len(totals)
    petal_length = 11.0
    petal_width = 3.8
    center_r = 3.2
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2

    for i, row in totals.iterrows():
        angle = angles[i]
        color = group_color[row["group"]]
        cx = (petal_length / 2 - 2.0) * np.cos(angle)
        cy = (petal_length / 2 - 2.0) * np.sin(angle)

        ellipse = Ellipse(
            (cx, cy), width=petal_length, height=petal_width,
            angle=np.degrees(angle), color=color, alpha=0.58, zorder=2
        )
        ax.add_patch(ellipse)
        ellipse_edge = Ellipse(
            (cx, cy), width=petal_length, height=petal_width,
            angle=np.degrees(angle), fill=False, color="white", lw=1.6, zorder=2.5
        )
        ax.add_patch(ellipse_edge)

        lx = (petal_length - 2.8) * np.cos(angle)
        ly = (petal_length - 2.8) * np.sin(angle)
        label = f"{row['letter']}\n{int(row['count'])}\n({row['pct']:.1f}%)"
        ax.text(
            lx, ly, label, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color="black", zorder=5
        )

    circle = plt.Circle((0, 0), center_r, color="white", zorder=4, ec="#333333", lw=1.8)
    ax.add_patch(circle)
    ax.text(
        0, 0, "17\nMAGs", ha="center", va="center",
        fontsize=19, fontweight="bold", color="#222222", zorder=6
    )
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)


def main():
    print("1. Saving standalone panels for Figure 1...")
    # Standalone Panel 1A
    fig_a = plt.figure(figsize=(10.5, 6.2))
    gs_a = gridspec.GridSpec(1, 2, width_ratios=[2.7, 1.3], wspace=0.05)
    draw_panel_a(fig_a.add_subplot(gs_a[0]), fig_a.add_subplot(gs_a[1]))
    fig_a.savefig(OUT_PNL / "Fig1A_MAG_Phylum_StackedBar.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig_a)

    # Standalone Panel 1C
    fig_c, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    draw_panel_c(ax_c1, ax_c2)
    fig_c.tight_layout()
    fig_c.savefig(OUT_PNL / "Fig1C_HQ17_QC_Scatter.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig_c)

    # Standalone Panel 1D
    fig_d, ax_d = plt.subplots(figsize=(7.5, 7.5))
    draw_panel_d(ax_d)
    fig_d.savefig(OUT_PNL / "Fig1D_COG_EqualPetal_17MAGs.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig_d)

    # Load Panel 1B from Figures/Panels/Fig1B_Phylogeny_89bac_with_legend.png
    panel_b_saved = OUT_PNL / "Fig1B_Phylogeny_89bac_with_legend.png"
    if not panel_b_saved.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "01b_Figure1B_Phylogeny_89bac.py")], check=True)
        alt_b = OUT_PNL / "Fig1B_Phylogeny_89MAGs.png"
        if alt_b.exists() and not panel_b_saved.exists():
            panel_b_saved = alt_b
    panel_b_crop = np.array(Image.open(panel_b_saved).convert("RGB"))

    print("2. Building 600 DPI composite Figure 1 (with Symptomatic/Asymptomatic in Panel C)...")
    fig = plt.figure(figsize=(21.0, 13.0))

    # Top row: Panel A (left) and Panel B (right)
    ax_a_bar = fig.add_axes([0.045, 0.51, 0.365, 0.455])
    ax_a_leg = fig.add_axes([0.420, 0.51, 0.155, 0.455])
    draw_panel_a(ax_a_bar, ax_a_leg)

    ax_b = fig.add_axes([0.575, 0.485, 0.415, 0.505])
    ax_b.imshow(panel_b_crop, interpolation="lanczos")
    ax_b.axis("off")

    # Bottom row: Panel C (left, 2 subplots) and Panel D (right, petal plot)
    ax_c_host   = fig.add_axes([0.045, 0.055, 0.305, 0.385])
    ax_c_status = fig.add_axes([0.395, 0.055, 0.305, 0.385])
    draw_panel_c(ax_c_host, ax_c_status)

    ax_d_comp = fig.add_axes([0.715, 0.025, 0.275, 0.445])
    draw_panel_d(ax_d_comp)

    # Panel labels A, B, C, D
    fig.text(0.008, 0.985, "A", fontsize=20, fontweight="bold", va="top", ha="left")
    fig.text(0.580, 0.985, "B", fontsize=20, fontweight="bold", va="top", ha="left")
    fig.text(0.008, 0.460, "C", fontsize=20, fontweight="bold", va="top", ha="left")
    fig.text(0.725, 0.460, "D", fontsize=20, fontweight="bold", va="top", ha="left")

    out_png = OUT_FIG / "Fig1.png"
    out_tif = OUT_FIG / "Fig1.tif"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    save_pace_tiff(out_png, out_tif)
    print(f"Saved: {out_png} and {out_tif}")



if __name__ == "__main__":
    main()
