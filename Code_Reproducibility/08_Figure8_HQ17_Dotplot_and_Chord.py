"""
08_Figure8_HQ17_Dotplot_and_Chord.py
====================================
Reproduces Figure 8 (Functional Dot-Plot & 3-Column Bipartite Chord Diagram of the 17 HQ MAGs):
- Panel A: Functional category dot-plot across the 17 HQ MAGs ordered by bacterial phylum.
- Panel B: 3-column bipartite chord diagram linking the top 30 marker genes (10 per major functional group,
  with outgoing link counts '(n)' indicating the number of distinct HQ MAGs encoding each marker term)
  to the 17 HQ MAGs and their corresponding functional categories.
"""

from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Patch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Arial"

PKG_DIR = Path(__file__).resolve().parent.parent
RES = PKG_DIR / "Source_Data" / "results"
OUT_FIG_DIR = PKG_DIR / "Figures"
OUT_PANEL_DIR = OUT_FIG_DIR / "Panels"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PANEL_DIR.mkdir(parents=True, exist_ok=True)

GROUP_COLORS = {
    "Community metabolism & nutrient cycling": "#1F77B4",        # blue
    "Extracellular enzymes & PGPR-associated traits": "#D62728", # red
    "Microbe\u2013plant interaction": "#2CA02C",                 # green
}
GROUP_ORDER = list(GROUP_COLORS.keys())

PHYLUM_COLORS = {
    "Acidobacteriota": "#4DAF4A",
    "Actinomycetota": "#377EB8",
    "Bacteroidota": "#984EA3",
    "Gemmatimonadota": "#F781BF",
    "Pseudomonadota": "#E41A1C",
    "Verrucomicrobiota": "#A65628",
    "Other": "#BDBDBD",
}
_pal_file = Path(__file__).parent / "Phylum_Color_Palette.tsv"
if _pal_file.exists():
    _pal_df = pd.read_csv(_pal_file, sep="\t")
    PHYLUM_COLORS.update(dict(zip(_pal_df["phylum"], _pal_df["color"])))


IAA_OLD = "IAA/auxin-related metabolism"
IAA_NEW = "Tryptophan metabolism (IAA precursor/pathway genes)"
IAA_NEW_MULTILINE_B = "Tryptophan metabolism\n(IAA precursor/pathway genes)"

N_PER_GROUP = 10
GAP_SLOTS = 2
MAG_COL = "#8E72B8"


def norm_mag(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"^Sg_", "", regex=True)
        .str.replace(r"\.fa$", "", regex=True)
        .str.strip()
    )


def load_data():
    src = pd.read_csv(RES / "Chord_top10_MAG_specific_gene_links_HQ17_corrected.tsv", sep="\t")
    src["mag_id"] = norm_mag(src["mag_id"])
    src["mag_label"] = src["mag_id"]
    src["functional_group"] = src["functional_group"].str.replace("\u2014", "\u2013", regex=False)
    src["functional_category"] = src["functional_category"].replace({IAA_OLD: IAA_NEW})

    meta = pd.read_csv(RES / "Metadata_17HQ_MAGs.tsv", sep="\t")
    meta["mag_id"] = norm_mag(meta["mag"])
    if "phylum" not in meta.columns:
        meta["phylum"] = "Other"
    meta = meta[["mag_id", "phylum"]].drop_duplicates("mag_id")
    return src, meta


def draw_panel_a(ax, src: pd.DataFrame, meta: pd.DataFrame, include_legends: bool = False):
    agg = (
        src.groupby(["mag_id", "functional_group", "functional_category"])["n_hits"]
        .sum()
        .reset_index(name="gene_count")
    )
    top_desc = (
        agg.groupby("functional_category")["gene_count"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .index.tolist()
    )
    df = agg[agg["functional_category"].isin(top_desc)].copy()

    mags_meta = (
        df[["mag_id"]].drop_duplicates()
        .merge(meta, on="mag_id", how="left")
        .fillna({"phylum": "Other"})
        .sort_values(["phylum", "mag_id"])
    )
    mags_ord = mags_meta["mag_id"].tolist()

    desc_grp = df.groupby("functional_category")["functional_group"].first().to_dict()
    cat_total = df.groupby("functional_category")["gene_count"].sum()
    cat_order_parts = []
    for grp in GROUP_ORDER:
        cats_in_grp = [d for d in top_desc if desc_grp.get(d) == grp]
        cats_sorted = sorted(cats_in_grp, key=lambda c: -cat_total.get(c, 0))
        cat_order_parts.extend(cats_sorted)
    remaining = [d for d in top_desc if d not in cat_order_parts]
    desc_ord = cat_order_parts + remaining

    mag_idx = {m: i for i, m in enumerate(mags_ord)}
    desc_idx = {d: i for i, d in enumerate(desc_ord)}

    df_plot = (
        df.sort_values("gene_count", ascending=False)
        .groupby(["mag_id", "functional_category"], as_index=False)
        .agg(gene_count=("gene_count", "sum"), functional_group=("functional_group", "first"))
    )
    max_count = int(df_plot["gene_count"].max())

    for _, row in df_plot.iterrows():
        xi = desc_idx.get(row["functional_category"])
        yi = mag_idx.get(row["mag_id"])
        if xi is None or yi is None:
            continue
        grp = row["functional_group"]
        col = GROUP_COLORS.get(grp, "#AAAAAA")
        sz = (row["gene_count"] / max_count) * 860 + 42
        ax.scatter(
            xi, yi, s=sz, c=col, alpha=0.84,
            edgecolors="#333333", linewidths=0.55, zorder=3
        )

    prev_grp = None
    stripe_start = None
    stripe_rects = []
    for xi, cat in enumerate(desc_ord):
        grp = desc_grp.get(cat, "")
        if grp != prev_grp:
            if prev_grp is not None and stripe_start is not None:
                stripe_rects.append((prev_grp, stripe_start, xi - 1))
            stripe_start = xi
            prev_grp = grp
    if prev_grp is not None and stripe_start is not None:
        stripe_rects.append((prev_grp, stripe_start, len(desc_ord) - 1))

    for grp, x0, x1 in stripe_rects:
        col = GROUP_COLORS.get(grp, "#AAAAAA")
        ax.axvspan(x0 - 0.5, x1 + 0.5, alpha=0.06, color=col, zorder=0)

    ax.set_xticks(range(len(desc_ord)))
    # Full untruncated category names on x-axis (wrap IAA_NEW on 2 lines for compact bottom margin)
    xtick_lbls = [IAA_NEW_MULTILINE_B if d == IAA_NEW else d for d in desc_ord]
    ax.set_xticklabels(xtick_lbls, rotation=42, ha="right", fontsize=16.5)
    for tick, cat in zip(ax.get_xticklabels(), desc_ord):
        grp = desc_grp.get(cat, "")
        col = GROUP_COLORS.get(grp, "#444444")
        tick.set_color(col)
        tick.set_fontweight("bold")

    ax.set_yticks(range(len(mags_ord)))
    ax.set_yticklabels(mags_ord, fontsize=17.5, fontweight="bold")
    ax.tick_params(axis="y", pad=8)

    ax.set_xlim(-0.6, len(desc_ord) - 0.4)
    ax.set_ylim(-0.8, len(mags_ord) - 0.2)
    ax.grid(color="#e6e6e6", zorder=1)
    ax.set_axisbelow(True)
    ax.set_ylabel("MAG (sorted by phylum)", fontsize=19, fontweight="bold", labelpad=10)

    phylum_map = meta.set_index("mag_id")["phylum"].to_dict()
    for i, m in enumerate(mags_ord):
        phy = phylum_map.get(m, "Other")
        pcol = PHYLUM_COLORS.get(phy, "#999999")
        ax.barh(
            i, 0.24, left=-0.76, height=0.74,
            color=pcol, alpha=0.90, zorder=4, clip_on=False
        )

    seen_phyla = []
    for m in mags_ord:
        phy = phylum_map.get(m, "Other")
        if phy not in seen_phyla:
            seen_phyla.append(phy)

    if include_legends:
        group_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=g, linewidth=0) for g in GROUP_ORDER]
        size_handles = [
            ax.scatter([], [], s=int(f * 860 + 42), c="#888888", alpha=0.65,
                       edgecolors="#555555", linewidths=0.4, label=f"{int(f * max_count)} hits")
            for f in [0.20, 0.50, 1.00]
        ]
        phy_patches = [mpatches.Patch(color=PHYLUM_COLORS.get(p, "#999999"), label=p) for p in seen_phyla]
        leg1 = ax.legend(handles=group_patches, fontsize=14, title="Functional group", title_fontsize=14.5,
                         loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.95)
        ax.add_artist(leg1)
        leg2 = ax.legend(handles=size_handles, fontsize=14, title="Total hit count", title_fontsize=14.5,
                         loc="center left", bbox_to_anchor=(1.01, 0.45), framealpha=0.95)
        ax.add_artist(leg2)
        ax.legend(handles=phy_patches, prop={"size": 13.5, "style": "normal"}, title="Phylum", title_fontsize=14.5,
                  loc="lower left", bbox_to_anchor=(1.01, 0.0), framealpha=0.95)

    return max_count, seen_phyla


def draw_panel_b(ax, src: pd.DataFrame, max_count_a: int, seen_phyla: list[str], show_combined_legends: bool = True):
    gene_totals = (
        src.groupby(["functional_group", "gene_display"], as_index=False)
        .agg(n_MAGs=("mag_label", "nunique"), total_hits=("n_hits", "sum"))
        .sort_values(["functional_group", "n_MAGs", "total_hits"], ascending=[True, False, False])
    )
    sel_genes = gene_totals.groupby("functional_group", group_keys=False).head(N_PER_GROUP).copy()
    sel_gene_set = set(zip(sel_genes["functional_group"], sel_genes["gene_display"]))
    gene_grp_map = dict(zip(sel_genes["gene_display"], sel_genes["functional_group"]))
    gene_nmag_map = dict(zip(sel_genes["gene_display"], sel_genes["n_MAGs"]))

    block = src[
        src.apply(lambda r: (r["functional_group"], r["gene_display"]) in sel_gene_set, axis=1)
    ].copy()

    genes_by_grp: dict[str, list[str]] = {
        grp: sel_genes[sel_genes["functional_group"] == grp]["gene_display"].tolist()
        for grp in GROUP_ORDER
    }

    mag_order = (
        block.groupby("mag_label")["n_hits"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    n_mags = len(mag_order)

    desc_raw = (
        block.groupby(["functional_group", "functional_category"], as_index=False)
        .agg(tot=("n_hits", "sum"))
        .sort_values(["functional_group", "tot"], ascending=[True, False])
    )
    desc_grp_map = dict(zip(desc_raw["functional_category"], desc_raw["functional_group"]))
    desc_order: list[str] = []
    for grp in GROUP_ORDER:
        sub = desc_raw[desc_raw["functional_group"] == grp]
        desc_order.extend(sub["functional_category"].tolist())
    n_descs = len(desc_order)

    mag_cat = (
        block.groupby(["mag_label", "functional_category", "functional_group"], as_index=False)
        .agg(cat_hits=("n_hits", "sum"))
    )

    slots: list[tuple[str, str] | None] = []
    for i, grp in enumerate(GROUP_ORDER):
        for g in genes_by_grp.get(grp, []):
            slots.append((g, grp))
        if i < len(GROUP_ORDER) - 1:
            for _ in range(GAP_SLOTS):
                slots.append(None)
    n_slots = len(slots)

    ax.set_xlim(-0.04, 1.15)
    ax.set_ylim(-0.02, 1.03)
    ax.axis("off")

    X_GENE = 0.17
    X_MAG = 0.45
    X_DESC = 0.72
    X_GENE_LBL = X_GENE - 0.013
    X_DESC_LBL = X_DESC + 0.013

    slot_y = [1.0 - (i + 1) / (n_slots + 1) for i in range(n_slots)]
    gene_y: dict[str, float] = {}
    for i, slot in enumerate(slots):
        if slot is not None:
            g, _ = slot
            gene_y[g] = slot_y[i]

    mag_y = {m: 1.0 - (i + 1) / (n_mags + 1) for i, m in enumerate(mag_order)}
    desc_y = {d: 1.0 - (i + 1) / (n_descs + 1) for i, d in enumerate(desc_order)}

    # Bezier links: gene -> MAG
    max_hits = block["n_hits"].max() if len(block) > 0 else 1
    for _, row in block.iterrows():
        g, m = row["gene_display"], row["mag_label"]
        if g not in gene_y or m not in mag_y:
            continue
        gy, my = gene_y[g], mag_y[m]
        col = GROUP_COLORS.get(gene_grp_map.get(g, GROUP_ORDER[0]), "#888888")
        lw = 0.45 + 2.5 * row["n_hits"] / max_hits
        cx1, cx2 = X_GENE + 0.075, X_MAG - 0.048
        verts = [(X_GENE, gy), (cx1, gy), (cx2, my), (X_MAG, my)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        ax.add_patch(PathPatch(
            MplPath(verts, codes),
            facecolor="none", edgecolor=col, lw=lw, alpha=0.27, zorder=1
        ))

    # Bezier links: MAG -> description
    max_cat = mag_cat["cat_hits"].max() if len(mag_cat) > 0 else 1
    for _, row in mag_cat.iterrows():
        m, d = row["mag_label"], row["functional_category"]
        if m not in mag_y or d not in desc_y:
            continue
        my, dy = mag_y[m], desc_y[d]
        col = GROUP_COLORS.get(row["functional_group"], "#888888")
        lw = 0.35 + 2.2 * row["cat_hits"] / max_cat
        cx1, cx2 = X_MAG + 0.048, X_DESC - 0.048
        verts = [(X_MAG, my), (cx1, my), (cx2, dy), (X_DESC, dy)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        ax.add_patch(PathPatch(
            MplPath(verts, codes),
            facecolor="none", edgecolor=col, lw=lw, alpha=0.23, zorder=1
        ))

    # Draw gene nodes (LEFT column) with (n) outgoing MAG links
    for g, y in gene_y.items():
        grp = gene_grp_map.get(g, GROUP_ORDER[0])
        col = GROUP_COLORS.get(grp, "#888888")
        n_m = gene_nmag_map.get(g, 0)
        ax.add_patch(plt.Rectangle(
            (X_GENE - 0.010, y - 0.0065), 0.020, 0.013,
            color=col, ec="black", lw=0.45, zorder=4
        ))
        lbl_str = f"{g} ({n_m})"
        ax.text(
            X_GENE_LBL, y, lbl_str, ha="right", va="center",
            fontsize=17.5, fontstyle="italic", color="#1a1a1a", zorder=6
        )

    # Draw MAG nodes (MIDDLE column)
    for m, y in mag_y.items():
        ax.add_patch(plt.Rectangle(
            (X_MAG - 0.038, y - 0.0125), 0.076, 0.025,
            color=MAG_COL, ec="black", lw=0.7, zorder=4
        ))
        ax.text(
            X_MAG, y, m, ha="center", va="center",
            fontsize=17.5, fontweight="bold", color="white", zorder=6
        )

    # Draw description nodes (RIGHT column) - NO truncation!
    for d, y in desc_y.items():
        grp = desc_grp_map.get(d, GROUP_ORDER[0])
        col = GROUP_COLORS.get(grp, "#888888")
        ax.add_patch(plt.Rectangle(
            (X_DESC - 0.010, y - 0.0065), 0.020, 0.013,
            color=col, ec="black", lw=0.45, zorder=4
        ))
        lbl = IAA_NEW_MULTILINE_B if d == IAA_NEW else d
        ax.text(
            X_DESC_LBL, y, lbl, ha="left", va="center",
            fontsize=17.0, color=col, fontweight="bold", zorder=6
        )

    # Column headers
    top_y = 1.0 - 1.0 / (n_slots + 1)
    ax.text(
        X_GENE - 0.02, top_y + 0.038, "Marker genes (top 30; n = linked MAGs)",
        ha="center", va="bottom", fontsize=18.5, fontweight="bold", color="#222222"
    )
    ax.text(
        X_MAG, 1.0 - 1.0 / (n_mags + 1) + 0.038, "17 HQ MAGs",
        ha="center", va="bottom", fontsize=18.5, fontweight="bold", color="#222222"
    )
    ax.text(
        X_DESC + 0.08, 1.0 - 1.0 / (n_descs + 1) + 0.038, "Functional description",
        ha="center", va="bottom", fontsize=18.5, fontweight="bold", color="#222222"
    )

    if show_combined_legends:
        # 1. Phylum legend (bottom-left of Panel B)
        phy_handles = [
            Patch(facecolor=PHYLUM_COLORS.get(p, "#999999"), edgecolor="none", label=p)
            for p in sorted(seen_phyla)
        ]
        leg_phy = ax.legend(
            handles=phy_handles,
            loc="upper left",
            bbox_to_anchor=(-0.03, -0.01),
            title="Phylum",
            title_fontsize=16.5,
            prop={"family": "Arial", "style": "normal", "size": 15.5},
            framealpha=0.95,
            edgecolor="#CCCCCC",
            borderpad=0.9,
            labelspacing=0.65,
        )
        leg_phy.get_title().set_fontweight("bold")
        ax.add_artist(leg_phy)

        # 2. Functional group / node type legend (top-right under Panel B)
        grp_handles = [
            Patch(facecolor=v, edgecolor="black", label=k, linewidth=0.7)
            for k, v in GROUP_COLORS.items()
        ]
        grp_handles.append(Patch(facecolor=MAG_COL, edgecolor="black", label="MAG node", linewidth=0.7))
        leg_grp = ax.legend(
            handles=grp_handles,
            loc="upper left",
            bbox_to_anchor=(0.24, -0.01),
            ncol=2,
            fontsize=15.5,
            title="Functional group / node type",
            title_fontsize=16.5,
            framealpha=0.95,
            edgecolor="#CCCCCC",
            borderpad=0.9,
            labelspacing=0.7,
        )
        leg_grp.get_title().set_fontweight("bold")
        ax.add_artist(leg_grp)

        # 3. Total hit count legend (below Functional group legend)
        size_handles = [
            ax.scatter([], [], s=int(f * 860 + 42), c="#888888", alpha=0.65,
                       edgecolors="#555555", linewidths=0.4, label=f"{int(f * max_count_a)} hits")
            for f in [0.20, 0.50, 1.00]
        ]
        leg_sz = ax.legend(
            handles=size_handles,
            loc="upper left",
            bbox_to_anchor=(0.24, -0.145),
            ncol=3,
            fontsize=15.5,
            title="Total hit count (Panel A bubble size)",
            title_fontsize=16.0,
            framealpha=0.95,
            edgecolor="#CCCCCC",
            borderpad=0.9,
            scatterpoints=1,
            columnspacing=1.5,
        )
        leg_sz.get_title().set_fontweight("bold")


def main():
    src, meta = load_data()

    # 1. Save standalone Panel A
    fig_a, ax_a = plt.subplots(figsize=(20, 14))
    max_count_a, seen_phyla = draw_panel_a(ax_a, src, meta, include_legends=True)
    fig_a.tight_layout(rect=[0, 0, 0.80, 1])
    fig_a.savefig(OUT_PANEL_DIR / "Fig8A_HQ17_Functional_Dotplot.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig_a)

    # 2. Save standalone Panel B
    fig_b, ax_b = plt.subplots(figsize=(24, 18))
    draw_panel_b(ax_b, src, max_count_a, seen_phyla, show_combined_legends=True)
    fig_b.savefig(OUT_PANEL_DIR / "Fig8B_HQ17_Chord_30Markers.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig_b)

    # 3. Build Combined 2-Panel Figure 8
    fig = plt.figure(figsize=(36, 18.5), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.28], wspace=0.06)

    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    max_count_a, seen_phyla = draw_panel_a(ax_left, src, meta, include_legends=False)
    draw_panel_b(ax_right, src, max_count_a, seen_phyla, show_combined_legends=True)

    # Panel labels A and B
    ax_left.text(
        -0.08, 1.03, "A", transform=ax_left.transAxes,
        fontsize=31, fontweight="bold", fontfamily="Arial", va="top", ha="left"
    )
    ax_right.text(
        0.98, 1.03, "B", transform=ax_right.transAxes,
        fontsize=31, fontweight="bold", fontfamily="Arial", va="top", ha="right"
    )

    png_path = OUT_FIG_DIR / "Fig8.png"
    tif_path = OUT_FIG_DIR / "Fig8.tif"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    target_scale = min(4500 / w, 5250 / h, 1.0)
    if target_scale < 1.0:
        img = img.resize((int(round(w * target_scale)), int(round(h * target_scale))), Image.Resampling.LANCZOS)
        img.save(png_path, format="PNG", dpi=(600, 600), optimize=True)
    img.save(tif_path, format="TIFF", dpi=(600, 600), compression="tiff_lzw")
    print(f"Saved: {png_path}, {tif_path} ({img.size[0]}x{img.size[1]} px)")


if __name__ == "__main__":
    main()

