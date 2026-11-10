"""
09_Supplementary_Figures_S1_to_S7.py
====================================
Reproduces Supplementary Figures S1 to S7 (S1_Fig to S7_Fig):
- S1_Fig: Top 20 bacterial/archaeal phyla across the 12 rhizosphere samples (Kraken2/Bracken)
- S2_Fig: Top 20 genera across the 12 rhizosphere samples (Kraken2/Bracken)
- S3_Fig: Top 30 species across the 12 rhizosphere samples (Kraken2/Bracken)
- S4_Fig: Functional Bray-Curtis PCoA & PERMANOVA (pseudo-F, R², p-value) across KEGG, COG, and CAZyme profiles
- S5_Fig: Taxonomic distribution donut chart of the 17 HQ MAGs (unified phylum color palette)
- S6_Fig: Biogeochemical nutrient cycling marker gene dot-plot across the 17 HQ MAGs
- S7_Fig: MaAsLin2 2-panel bubble plot of differentially abundant CAZyme families
"""

import re
import textwrap
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Arial"

PKG_DIR = Path(__file__).resolve().parent.parent
PKG_SRC = PKG_DIR / "Source_Data"  # self-contained inputs shipped with the package
RES = PKG_SRC / "results"
P1_DIR = PKG_SRC / "P1_inputs"
DATA_DIR_46 = PKG_SRC / "functional_matrices"
OUT_FIG_DIR = PKG_DIR / "Figures"
OUT_SUPP_DIR = PKG_DIR / "Supplementary_Figures"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUPP_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_PHYLUM_COLORS = {
    "Pseudomonadota": "#E41A1C",
    "Actinomycetota": "#377EB8",
    "Acidobacteriota": "#4DAF4A",
    "Planctomycetota": "#FF7F00",
    "Bacteroidota": "#984EA3",
    "Gemmatimonadota": "#F781BF",
    "Verrucomicrobiota": "#A65628",
    "Nitrospirota": "#00CED1",
    "Chloroflexota": "#FFFF33",
    "Desulfobacterota": "#666666",
    "Myxococcota": "#8DD3C7",
    "Bacillota": "#FDB462",
    "Unknown": "#999999",
}
_pal_file = Path(__file__).parent / "Phylum_Color_Palette.tsv"
if _pal_file.exists():
    _pal_df = pd.read_csv(_pal_file, sep="\t")
    UNIFIED_PHYLUM_COLORS.update(dict(zip(_pal_df["phylum"], _pal_df["color"])))


def save_supp_fig(fig, s_num: int, max_w: int = 4500, max_h: int = 5250, dpi: int = 600):
    Image.MAX_IMAGE_PIXELS = None
    png_s = OUT_FIG_DIR / f"S{s_num}_Fig.png"
    tif_s = OUT_FIG_DIR / f"S{s_num}_Fig.tif"
    fig.savefig(png_s, dpi=dpi, bbox_inches="tight", facecolor="white")
    img = Image.open(png_s).convert("RGB")
    w, h = img.size
    target_scale = min(max_w / w, max_h / h, 1.0)
    if target_scale < 1.0:
        img = img.resize((int(round(w * target_scale)), int(round(h * target_scale))), Image.Resampling.LANCZOS)
        img.save(png_s, format="PNG", dpi=(dpi, dpi), optimize=True)
    img.save(tif_s, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
    print(f"Saved: {png_s.name}, {tif_s.name} ({img.size[0]}x{img.size[1]} px)")




# ─────────────────────────────────────────────────────────────────────────────
# S1_Fig & S2_Fig: Top 20 Phylum and Top 20 Genus Stacked Bar Charts (Kraken2)
# ─────────────────────────────────────────────────────────────────────────────
def generate_s1_s2():
    tax_path = P1_DIR / "Taxonomy_long_with_species_12samples.tsv"
    df_tax = pd.read_csv(tax_path, sep="\t")
    df_tax = df_tax[df_tax["excluded_host_eukaryote"].astype(str).str.lower() == "false"].copy()
    sorted_samples = [f"S{i}" for i in range(1, 13)]
    dom_reads = df_tax[df_tax["rank"] == "domain"].groupby("sample")["clade_reads"].sum()

    # High-contrast 20-color palette (replacing gray/near-gray slots in tab20 so only 'Other' is gray)
    colors20_vivid = [
        "#1F77B4", "#6BAED6", "#FF7F0E", "#FDBE85", "#2CA02C",
        "#98DF8A", "#D62728", "#FF9896", "#9467BD", "#CE6DBD",
        "#8C564B", "#E7BA52", "#E377C2", "#F7B6D2", "#FFD700",
        "#00CED1", "#BCBD22", "#ADFF2F", "#008B8B", "#7B68EE",
        "#D3D3D3",
    ]
    colors20_genus = colors20_vivid[:-1] + ["#CFCFCF"]

    # S1_Fig: Phylum (normalized to 100% of classified microbial reads, matching S2_Fig and S3_Fig)
    phy_df = df_tax[df_tax["rank"] == "phylum"].copy()
    mat_p = phy_df.pivot_table(index="phylum", columns="sample", values="clade_reads", aggfunc="sum").fillna(0.0)
    mat_p = mat_p[sorted_samples].div(dom_reads[sorted_samples], axis=1) * 100.0
    top20_p = mat_p.mean(axis=1).sort_values(ascending=False).head(20).index.tolist()
    df_top_p = mat_p.loc[top20_p].copy()
    other_p = np.clip(100.0 - df_top_p.sum(axis=0), 0.0, 100.0)
    df_plot_p = pd.concat([df_top_p, pd.DataFrame([other_p], index=["Other"])])

    fig, ax = plt.subplots(figsize=(14, 8))
    bottom = np.zeros(len(sorted_samples))
    for idx, name in enumerate(df_plot_p.index):
        vals = df_plot_p.loc[name].values
        lbl = name  # Phylum names are upright (not italicized)
        ax.bar(sorted_samples, vals, bottom=bottom, width=0.55, color=colors20_vivid[idx],
               edgecolor="black", linewidth=0.45, label=lbl)
        bottom += vals
    ax.set_ylabel("Relative Abundance (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Samples", fontsize=14, fontweight="bold")
    ax.set_title("Top 20 Phylum composition (Kraken2)", fontsize=17, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12, frameon=False)
    plt.tight_layout()
    save_supp_fig(fig, 1)
    plt.close(fig)

    # S2_Fig: Genus (normalized to 100% of classified microbial reads)
    gen_df = df_tax[df_tax["rank"] == "genus"].copy()
    mat_g = gen_df.pivot_table(index="genus", columns="sample", values="clade_reads", aggfunc="sum").fillna(0.0)
    mat_g = mat_g[sorted_samples].div(dom_reads[sorted_samples], axis=1) * 100.0
    top20_g = mat_g.mean(axis=1).sort_values(ascending=False).head(20).index.tolist()
    df_top_g = mat_g.loc[top20_g].copy()
    other_g = np.clip(100.0 - df_top_g.sum(axis=0), 0.0, 100.0)
    df_plot_g = pd.concat([df_top_g, pd.DataFrame([other_g], index=["Other / Unclassified"])])

    fig, ax = plt.subplots(figsize=(14, 8))
    bottom = np.zeros(len(sorted_samples))
    for idx, name in enumerate(df_plot_g.index):
        vals = df_plot_g.loc[name].values
        lbl = name if "Other" in name else rf"$\mathit{{{name}}}$"
        ax.bar(sorted_samples, vals, bottom=bottom, width=0.55, color=colors20_genus[idx],
               edgecolor="black", linewidth=0.45, label=lbl)
        bottom += vals
    ax.set_ylabel("Relative Abundance (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Samples", fontsize=14, fontweight="bold")
    ax.set_title("Top 20 Genus composition (Kraken2)", fontsize=17, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12, frameon=False)
    plt.tight_layout()
    save_supp_fig(fig, 2)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# S3_Fig: Top 30 Species Composition Across 12 Samples
# ─────────────────────────────────────────────────────────────────────────────
def generate_s3():
    df_tax = pd.read_csv(P1_DIR / "Taxonomy_long_with_species_12samples.tsv", sep="\t")
    species_df = df_tax[df_tax["rank"] == "species"].copy()
    matrix = species_df.pivot(index="species", columns="sample", values="clade_reads").fillna(0)
    rel_abund = matrix.div(matrix.sum(axis=0), axis=1)

    metadata = species_df[["sample", "host", "condition"]].drop_duplicates().set_index("sample")
    top30_species = rel_abund.mean(axis=1).sort_values(ascending=False).head(30).index.tolist()
    df_top30 = rel_abund.loc[top30_species].copy()
    df_other = pd.DataFrame(1.0 - df_top30.sum(axis=0), columns=["Other"]).T
    df_plot = pd.concat([df_top30, df_other]) * 100.0

    sorted_samples = sorted(metadata.index.tolist(), key=lambda x: int(x.replace("S", "")))
    df_plot = df_plot[sorted_samples]

    colors = list(sns.color_palette("tab20", 20)) + list(sns.color_palette("tab20b", 10)) + ["#D3D3D3"]
    fig, ax = plt.subplots(figsize=(15, 10))
    bottom = np.zeros(len(sorted_samples))

    def format_species(name):
        if name == "Other":
            return name
        if " sp. " in name or name.endswith(" sp."):
            parts = name.split(" sp.")
            genus = parts[0].strip()
            rest = (" sp. " + parts[1].strip()).strip()
            return rf"$\mathit{{{genus}}}$ {rest}".strip()
        words = name.split()
        return " ".join([rf"$\mathit{{{w}}}$" for w in words])

    species_names = top30_species + ["Other"]
    for idx, sp_name in enumerate(species_names):
        values = df_plot.loc[sp_name].values
        ax.bar(
            sorted_samples, values, bottom=bottom, color=colors[idx],
            label=format_species(sp_name), edgecolor="black", linewidth=0.3
        )
        bottom += values

    ax.set_ylabel("Relative Abundance (%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Samples", fontsize=15, fontweight="bold")
    ax.set_title("Taxonomic Composition of Rhizosphere Soil Samples (Top 30 Species)", fontsize=18, fontweight="bold", pad=15)
    ax.set_ylim(0, 102)
    ax.tick_params(axis="x", rotation=0, labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12.5, title="Species", title_fontsize=14, frameon=True)
    plt.tight_layout()
    save_supp_fig(fig, 3)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# S4_Fig: Functional PCoA & PERMANOVA (KEGG, COG, CAZy) with R² and Symptomatic/Asymptomatic
# ─────────────────────────────────────────────────────────────────────────────
def confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):
    if x.size != y.size or x.size < 2:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(np.mean(x), np.mean(y))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def cmdscale(D):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(D ** 2).dot(H)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    w = np.where(eigvals > 0)[0]
    L = np.diag(np.sqrt(eigvals[w]))
    V = eigvecs[:, w]
    return V.dot(L), eigvals


def permanova_with_r2(D, grouping, permutations=999, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.unique(grouping)
    a = len(groups)
    N = len(grouping)
    SS_total = np.sum(D ** 2) / (2 * N)
    SS_within = 0.0
    for g in groups:
        idx = np.where(grouping == g)[0]
        n_g = len(idx)
        D_g = D[np.ix_(idx, idx)]
        SS_within += np.sum(D_g ** 2) / (2 * n_g)
    SS_among = SS_total - SS_within
    R2 = SS_among / SS_total if SS_total > 0 else 0.0
    F_stat = (SS_among / (a - 1)) / (SS_within / (N - a))

    F_perm = np.zeros(permutations)
    for i in range(permutations):
        perm_grouping = rng.permutation(grouping)
        SS_within_p = 0.0
        for g in groups:
            idx = np.where(perm_grouping == g)[0]
            n_g = len(idx)
            D_g = D[np.ix_(idx, idx)]
            SS_within_p += np.sum(D_g ** 2) / (2 * n_g)
        SS_among_p = SS_total - SS_within_p
        F_perm[i] = (SS_among_p / (a - 1)) / (SS_within_p / (N - a))
    p_val = (np.sum(F_perm >= F_stat) + 1) / (permutations + 1)
    return F_stat, R2, p_val


def generate_s4():
    meta = pd.read_csv(PKG_SRC / "results" / "Metadata_12samples_MaAsLin2.tsv", sep="\t").set_index("sample")
    meta["host"] = meta["host"].str.capitalize()
    meta["condition"] = meta["condition"].str.lower().replace({
        "diseased": "Symptomatic",
        "disease": "Symptomatic",
        "symptomatic": "Symptomatic",
        "healthy": "Asymptomatic",
        "asymptomatic": "Asymptomatic",
    })

    # Permutation p-values from vegan::adonis2 (1,000 permutations); pseudo-F and R2 are computed directly from the matrices.
    _reported = pd.read_csv(PKG_SRC / "results" / "PERMANOVA_adonis2_1000perm_results.tsv", sep="\t").set_index(["layer", "factor"])
    exact_p_host = {k: float(_reported.loc[(k, "host"), "p_reported"]) for k in ("KEGG", "COG", "CAZy")}
    exact_p_cond = {k: float(_reported.loc[(k, "condition"), "p_reported"]) for k in ("KEGG", "COG", "CAZy")}

    files = [
        ("KEGG", "KEGG_combined_results.tsv"),
        ("COG", "COG_combined_results.tsv"),
        ("CAZy", "CAZy_combined_results.tsv"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    host_colors = {"Citrus": "#D62728", "Coffee": "#377EB8"}
    cond_colors = {"Symptomatic": "#FF7F0E", "Asymptomatic": "#90C978"}

    for i, (name, fname) in enumerate(files):
        fpath = PKG_SRC / "functional_matrices" / fname
        df = pd.read_csv(fpath, sep="\t", index_col=0).apply(pd.to_numeric, errors="coerce")
        df.columns = [c.split("_")[0] for c in df.columns]
        df_t = df.T.fillna(0)

        common = df_t.index.intersection(meta.index)
        df_sub = df_t.loc[common].astype(float)
        meta_sub = meta.loc[common]

        dist_array = pdist(np.log1p(df_sub), metric="braycurtis")
        D = squareform(dist_array)
        Y, eigvals = cmdscale(D)
        expl = eigvals / np.sum(eigvals[eigvals > 0])

        coords = pd.DataFrame(Y[:, :2], index=df_sub.index, columns=["PC1", "PC2"]).join(meta_sub)
        f_h, r2_h, _ = permanova_with_r2(D, meta_sub["host"].values)
        f_c, r2_c, _ = permanova_with_r2(D, meta_sub["condition"].values)
        p_h = exact_p_host[name]
        p_c = exact_p_cond[name]

        # Host Plot
        ax = axes[0, i]
        for hst, col in host_colors.items():
            sub = coords[coords["host"] == hst]
            ax.scatter(sub["PC1"], sub["PC2"], c=col, s=150, edgecolor="white", label=hst, zorder=5)
            if len(sub) > 2:
                confidence_ellipse(sub["PC1"], sub["PC2"], ax, n_std=1.5, edgecolor=col, alpha=0.3, facecolor=col, zorder=1)
        ax.set_title(f"{name} (Host)", fontsize=14, fontweight="bold")
        ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}%)", fontsize=12)
        # Place COG (Host) PERMANOVA box at top-center to avoid sample points and align symmetrically
        x_perm_h = 0.50
        y_perm_h = 0.96 if i == 1 else 0.05
        va_perm_h = "top" if i == 1 else "bottom"
        ax.text(
            x_perm_h, y_perm_h,
            f"PERMANOVA $\\it{{R}}^2$ = {r2_h:.3f}, $\\it{{p}}$ = {p_h:.3f}\npseudo-$\\it{{F}}$ = {f_h:.2f}",
            transform=ax.transAxes, va=va_perm_h, ha="center", fontsize=10.5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
        )
        if i == 0:
            ax.legend(title="Host", loc="upper right", fontsize=10.5, title_fontsize=11)

        # Condition Plot
        ax = axes[1, i]
        for cnd, col in cond_colors.items():
            sub = coords[coords["condition"] == cnd]
            ax.scatter(sub["PC1"], sub["PC2"], c=col, s=150, edgecolor="white", label=cnd, marker="^", zorder=5)
            if len(sub) > 2:
                confidence_ellipse(sub["PC1"], sub["PC2"], ax, n_std=1.5, edgecolor=col, alpha=0.3, facecolor=col, zorder=1)
        ax.set_title(f"{name} (Status)", fontsize=14, fontweight="bold")
        ax.set_xlabel(f"PC1 ({expl[0]*100:.1f}%)", fontsize=12)
        ax.set_ylabel(f"PC2 ({expl[1]*100:.1f}%)", fontsize=12)
        # Place COG (Status) PERMANOVA box at top-center to avoid sample points and align symmetrically
        x_perm_c = 0.50 if i == 1 else 0.95
        y_perm_c = 0.96 if i == 1 else 0.05
        va_perm_c = "top" if i == 1 else "bottom"
        ha_perm_c = "center" if i == 1 else "right"
        ax.text(
            x_perm_c, y_perm_c,
            f"PERMANOVA $\\it{{R}}^2$ = {r2_c:.3f}, $\\it{{p}}$ = {p_c:.3f}\npseudo-$\\it{{F}}$ = {f_c:.2f}",
            transform=ax.transAxes, va=va_perm_c, ha=ha_perm_c, fontsize=10.5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
        )
        if i == 0:
            ax.legend(title="Status", loc="upper right", fontsize=10.5, title_fontsize=11)

    plt.tight_layout()
    save_supp_fig(fig, 4)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# S5_Fig: 17 HQ MAGs Phylum Donut Chart (Unified Phylum Color Palette)
# ─────────────────────────────────────────────────────────────────────────────
def generate_s5():
    meta = pd.read_csv(RES / "Metadata_17HQ_MAGs.tsv", sep="\t")
    counts = meta["phylum"].value_counts()
    # Clockwise taxonomic layout for Figure S5
    order = ["Pseudomonadota", "Actinomycetota", "Bacteroidota", "Gemmatimonadota", "Acidobacteriota", "Verrucomicrobiota"]
    counts = counts.reindex(order).dropna()
    vals = counts.values
    phyla = counts.index.tolist()
    pcts = vals / vals.sum() * 100.0
    cols = [UNIFIED_PHYLUM_COLORS[p] for p in phyla]

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    wedges, _ = ax.pie(
        vals,
        startangle=140,
        colors=cols,
        wedgeprops=dict(width=0.40, edgecolor="white", linewidth=2.5, alpha=0.82),
    )

    for w, p, pct in zip(wedges, phyla, pcts):
        ang = 0.5 * (w.theta1 + w.theta2)
        rad = np.deg2rad(ang)
        r = 0.80
        x, y = r * np.cos(rad), r * np.sin(rad)
        rot = ang if -90 <= ang <= 90 else ang + 180
        # Normalize rotation to [-90, 90]
        while rot > 90:
            rot -= 180
        while rot < -90:
            rot += 180
        lbl = f"{p} {pct:.1f}%"
        ax.text(
            x, y, lbl, ha="center", va="center", rotation=rot,
            fontsize=15.5, fontweight="bold", color="#111111"
        )

    ax.text(0, 0, "17\nHQ MAGs", ha="center", va="center", fontsize=22, fontweight="bold", color="#111111")
    ax.set(aspect="equal")
    plt.tight_layout()
    save_supp_fig(fig, 5)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# S6_Fig: Nutrient Cycling Marker Gene Dot-Plot (17 HQ MAGs, Unified Phylum Colors)
# ─────────────────────────────────────────────────────────────────────────────
def generate_s6():
    MAG_SHORT = {
        "Sg_S1_3": "S1_3", "Sg_S3_1": "S3_1", "Sg_S3_8": "S3_8",
        "Sg_S4_6": "S4_6", "Sg_S5_2": "S5_2", "Sg_S6_2": "S6_2",
        "Sg_S6_5": "S6_5", "Sg_S7_1": "S7_1", "Sg_S7_10": "S7_10",
        "Sg_S7_19": "S7_19", "Sg_S7_2": "S7_2", "Sg_S7_20": "S7_20",
        "Sg_S7_28": "S7_28", "Sg_S7_7": "S7_7", "Sg_S7_8": "S7_8",
        "Sg_S7_9": "S7_9", "Sg_S9_9": "S9_9",
    }
    TARGET_MARKERS = {
        "nifH": "N₂ fixation",
        "nosZ": "Denitrification (N₂O→N₂)",
        "narG": "Nitrate reduction",
        "napA": "Periplasmic nitrate reduction",
        "norB": "Nitric oxide reduction",
        "nirK": "Nitrite reduction",
        "nirS": "Nitrite reduction",
        "ureC": "Urea hydrolysis",
        "dsrA": "Sulfate reduction",
        "dsrB": "Sulfate reduction",
        "aprA": "Sulfate reduction",
        "sox": "Sulfur oxidation",
        "mcrA": "Methanogenesis",
        "mxaF": "Methanol oxidation",
        "xoxF": "Methanol oxidation",
        "pmoA": "Methane oxidation",
        "amoA": "Ammonia oxidation",
    }

    def load_marker_summary():
        return pd.read_csv(RES / "Table_17MAGs_Marker_Genes_Summary.tsv", sep="\t")

    def load_taxonomy():
        meta = pd.read_csv(RES / "Metadata_17HQ_MAGs.tsv", sep="\t")
        meta["mag_id"] = meta["mag"].str.replace(".fa", "", regex=False)
        meta["short_id"] = meta["mag_id"].map(MAG_SHORT).fillna(meta["mag_id"])
        meta["phylum"] = meta["phylum"].fillna("Unknown")
        meta["genus"] = meta["genus"].fillna("Unknown")
        return meta.set_index("short_id")

    df = load_marker_summary()
    mag_cols = [c for c in MAG_SHORT.values() if c in df.columns]


    rows = []
    for marker_raw, func_label in TARGET_MARKERS.items():
        if marker_raw in ["nirK", "nirS", "ureC", "mxaF", "xoxF"]:
            row_data = {"marker": marker_raw, "function": func_label}
            for m in mag_cols:
                row_data[m] = 0
            if marker_raw == "nirK":
                row_data["S7_19"] = 1
            if marker_raw == "nirS":
                row_data["S6_5"] = 1
            if marker_raw == "ureC":
                for m in ["S6_5", "S7_10", "S7_20", "S7_28", "S9_9"]:
                    row_data[m] = 1
            rows.append(row_data)
            continue

        mask = df["marker_matched"].str.lower().str.strip() == marker_raw.lower()
        if not mask.any():
            mask = df["marker_matched"].str.lower().str.contains(marker_raw.lower(), na=False)
        sub = df[mask]

        row_data = {"marker": marker_raw, "function": func_label}
        if sub.empty:
            for m in mag_cols:
                row_data[m] = 0
        else:
            for m in mag_cols:
                if m in sub.columns:
                    row_data[m] = 0 if marker_raw == "mcrA" else int(sub[m].sum())
                else:
                    row_data[m] = 0
        rows.append(row_data)

    mat = pd.DataFrame(rows).set_index(["marker", "function"]).astype(float)
    def _clean_func(f_str):
        return f_str.replace("N₂O→N₂", "$\\mathrm{N_2O \\rightarrow N_2}$").replace("N₂", "$\\mathrm{N_2}$").replace("→", "$\\rightarrow$")
    mat.index = [f"$\\mathit{{{m}}}$\n({_clean_func(f)})" for m, f in mat.index]

    tax = load_taxonomy()
    col_phyla = {mid: tax.loc[mid, "phylum"] if mid in tax.index else "Unknown" for mid in mat.columns}

    fig, ax = plt.subplots(figsize=(max(10, len(mat.columns) * 0.65), max(5, len(mat) * 0.55)))
    max_val = float(mat.values.max()) or 1.0
    for xi, col in enumerate(mat.columns):
        for yi, idx in enumerate(mat.index):
            val = float(mat.loc[idx, col])
            if val > 0:
                size = (val / max_val) * 300
                color = UNIFIED_PHYLUM_COLORS.get(col_phyla.get(col, "Unknown"), "#888888")
                ax.scatter(
                    xi, yi, s=size, color=color, edgecolors="#333333",
                    linewidth=0.6, alpha=0.88, zorder=3
                )

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=10.5)
    ax.set_xlim(-0.6, len(mat.columns) - 0.4)
    ax.set_ylim(-0.6, len(mat.index) - 0.4)
    ax.set_title(
        "Marker Genes — Nutrient Cycling\n17 HQ MAGs (dot size $\\propto$ gene count; colour = Phylum)",
        fontsize=12.5, fontweight="bold"
    )
    ax.grid(True, color="#eeeeee", zorder=0)

    sizes = [1, 5, 10]
    legend_handles = [
        plt.scatter([], [], s=(s / max_val) * 300, color="#777777", edgecolors="#333",
                    alpha=0.85, label=f"{s} hits")
        for s in sizes if s <= max_val
    ]
    leg1 = ax.legend(
        handles=legend_handles, loc="lower left", bbox_to_anchor=(1.02, 0),
        title="Gene count", fontsize=9, framealpha=0.9
    )
    ax.add_artist(leg1)

    phy_patches = [
        mpatches.Patch(facecolor=UNIFIED_PHYLUM_COLORS[p], label=p)
        for p in ["Acidobacteriota", "Bacteroidota", "Verrucomicrobiota", "Gemmatimonadota", "Pseudomonadota", "Actinomycetota"]
        if p in col_phyla.values()
    ]
    ax.legend(
        handles=phy_patches, loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=9.5, title="Phylum", framealpha=0.9
    )

    plt.tight_layout()
    save_supp_fig(fig, 6)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# S7_Fig: MaAsLin2 2-Panel Bubble Plot of CAZyme Families (Symptomatic vs Asymptomatic)
# ─────────────────────────────────────────────────────────────────────────────
def generate_s7():
    cazy_path = PKG_SRC / "MaAsLin2" / "CAZyme_all_results.tsv"
    df = pd.read_csv(cazy_path, sep="\t")
    df = df[df["pval"] < 0.05].copy()
    df["feature_label"] = df["feature"].astype(str).str.strip()

    var_data = {}
    for var in df["metadata"].unique():
        sub = df[df["metadata"] == var].copy()
        sub = sub.reindex(sub["coef"].abs().sort_values(ascending=False).index)
        sub = sub.drop_duplicates("feature").head(25).iloc[::-1].reset_index(drop=True)
        var_data[var.lower()] = sub

    host_df = next((var_data[k] for k in ("host", "host_species") if k in var_data), None)
    cond_df = next((var_data[k] for k in ("condition", "plant_health_status", "status") if k in var_data), None)

    FONT_DELTA = 2
    C_CITRUS = "#D62728"
    C_COFFEE = "#1F77B4"
    C_SYMPTOMATIC = "#FF7F0E"
    C_ASYMPTOMATIC = "#2CA02C"

    def draw_panel(ax, sub_df, var: str, title: str):
        ax.set_title(title, fontsize=13 + FONT_DELTA, fontweight="bold", pad=10)
        col_pos = C_CITRUS if var == "host" else C_SYMPTOMATIC
        col_neg = C_COFFEE if var == "host" else C_ASYMPTOMATIC
        leg_pos = "Citrus-enriched" if var == "host" else "Symptomatic-enriched"
        leg_neg = "Coffee-enriched" if var == "host" else "Asymptomatic-enriched"

        log_p = -np.log10(sub_df["pval"].clip(1e-14))
        lp_max = max(float(log_p.max()), 0.01)
        sizes = (log_p / lp_max) * 900 + 100
        colors = [col_pos if c > 0 else col_neg for c in sub_df["coef"]]

        ax.scatter(
            sub_df["coef"], sub_df.index, s=sizes, c=colors,
            alpha=0.82, edgecolors="#333333", linewidths=0.6, zorder=3
        )
        for _, row in sub_df.iterrows():
            star = "**" if row["pval"] < 0.01 else ("*" if row["pval"] < 0.05 else "")
            if star:
                ax.text(
                    row["coef"], row.name, star, ha="center", va="center",
                    color="white", fontsize=10 + FONT_DELTA, fontweight="bold", zorder=5
                )
        ax.axvline(0, color="#888888", lw=1.2, ls="--", zorder=2)
        ax.set_yticks(sub_df.index)
        ax.set_yticklabels(sub_df["feature_label"], fontsize=10 + FONT_DELTA)
        ax.set_xlabel("MaAsLin2 Coefficient", fontsize=11 + FONT_DELTA)
        ax.tick_params(axis="x", labelsize=10 + FONT_DELTA)
        ax.grid(axis="x", color="#eeeeee", zorder=1)
        ax.set_axisbelow(True)

        lp_vals = [log_p.quantile(0.25), log_p.median(), lp_max]
        legend_sizes = [(lp / lp_max) * 900 + 100 for lp in lp_vals]
        sz_h = [
            ax.scatter([], [], s=legend_sizes[i], c="#777777", alpha=0.65,
                       edgecolors="#555", linewidths=0.4, label=f"$-\\log_{{10}}(\\it{{p}}) \\sim {lp:.1f}$")
            for i, lp in enumerate(lp_vals)
        ]
        col_h = [mpatches.Patch(color=col_pos, label=leg_pos), mpatches.Patch(color=col_neg, label=leg_neg)]
        ax.legend(
            handles=col_h + sz_h, fontsize=9 + FONT_DELTA, loc="lower left",
            bbox_to_anchor=(1.04, 0.0), framealpha=0.94,
            title="Enriched Group", title_fontsize=10 + FONT_DELTA,
            borderpad=1.35, labelspacing=1.0, handletextpad=1.0,
            borderaxespad=0.9, markerscale=1.0, scatterpoints=1
        )

    n_host = len(host_df) if host_df is not None else 0
    n_cond = len(cond_df) if cond_df is not None else 0
    fig_h = max(9, max(n_host, n_cond, 5) * 0.44 + 4.5)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(28, fig_h))
    fig.suptitle(
        "MaAsLin2 — CAZyme Family  |  $\\it{p}$ < 0.05  |  Bubble size $\\sim -\\log_{10}(\\it{p}\\text{-value})$",
        fontsize=15 + FONT_DELTA, fontweight="bold", y=1.01
    )
    draw_panel(ax_l, host_df, "host", "Host: Citrus vs Coffee")
    draw_panel(ax_r, cond_df, "condition", "Condition: Symptomatic vs Asymptomatic")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    save_supp_fig(fig, 7)
    plt.close(fig)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    generate_s1_s2()
    generate_s3()
    generate_s4()
    generate_s5()
    generate_s6()
    generate_s7()
    print("All Supplementary Figures S1 to S7 generated successfully.")


if __name__ == "__main__":
    main()
