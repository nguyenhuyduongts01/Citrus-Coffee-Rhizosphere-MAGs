"""
02_Figure2_Alpha_Beta_Diversity.py
==================================
Reproduces Figure 2 (4-panel Alpha and Beta Diversity across 12 rhizosphere soil samples):
  - Panel A: Observed genus richness (Bracken genus count > 0) across Host (Citrus vs. Coffee) and Plant Health Status (Symptomatic vs. Asymptomatic)
  - Panel B: Shannon diversity index across Host (Citrus vs. Coffee) and Plant Health Status (Symptomatic vs. Asymptomatic)
  - Panel C: Bray-Curtis PCoA by Host (Citrus vs. Coffee) with PERMANOVA (adonis2) F, R², and p-value
  - Panel D: Bray-Curtis PCoA by Plant Health Status (Symptomatic vs. Asymptomatic) with PERMANOVA (adonis2) F, R², and p-value

Outputs:
  - Figures/Fig2.png (600 DPI)
  - Figures/Fig2.tif (600 DPI, LZW-compressed RGB TIFF, PACE-compliant)
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from scipy import linalg
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec

PKG_DIR = Path(__file__).resolve().parent.parent
RES  = PKG_DIR / "Source_Data" / "results"
OUT_FIG = PKG_DIR / "Figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

COLORS = {
    "citrus":      "#E44D3A",
    "coffee":      "#4B9CD3",
    "diseased":    "#F59E0B",
    "disease":     "#F59E0B",
    "symptomatic": "#F59E0B",
    "healthy":     "#66B032",
    "asymptomatic":"#66B032",
}

DISPLAY_LABELS = {
    "citrus":   "Citrus",
    "coffee":   "Coffee",
    "diseased": "Symptomatic",
    "disease":  "Symptomatic",
    "healthy":  "Asymptomatic",
}

MARKERS = {
    "citrus":   "s",
    "coffee":   "o",
    "diseased": "^",
    "disease":  "^",
    "healthy":  "o",
}

PERMANOVA = {
    "host": {
        "f": 3.746, "r2": 0.273, "p": 0.0030, "sig": "**",
        "method": "adonis2, Bray-Curtis, 1000 perms"
    },
    "condition": {
        "f": 1.563, "r2": 0.135, "p": 0.1180, "sig": "ns",
        "method": "adonis2, Bray-Curtis, 1000 perms"
    },
}


def save_pace_tiff(png_path: Path, tif_path: Path, max_w: int = 4500, max_h: int = 5250, dpi: int = 600):
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    target_scale = min(max_w / w, max_h / h, 1.0)
    if target_scale < 1.0:
        img = img.resize((int(round(w * target_scale)), int(round(h * target_scale))), Image.LANCZOS)
        img.save(png_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    img.save(tif_path, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))



def confidence_ellipse(x, y, ax, n_std=1.96, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def draw_alpha_subpanel(ax, df, metric, group_col, order, ylabel=None, ylim=None):
    data = [df.loc[df[group_col] == lvl, metric].values for lvl in order]
    bp = ax.boxplot(
        data, positions=[0, 1], widths=0.52, patch_artist=True,
        showfliers=False,
        whiskerprops={"color": "#222222", "lw": 1.1},
        capprops={"color": "#222222", "lw": 1.1},
        medianprops={"color": "black", "lw": 1.8},
        boxprops={"edgecolor": "#222222", "lw": 0.9},
    )
    for patch, lvl in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS[lvl])
        patch.set_alpha(0.78)

    np.random.seed(42)
    for xpos, lvl, vals in zip([0, 1], order, data):
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(vals), xpos) + jitter, vals,
            s=46, c=COLORS[lvl], edgecolors="#222222", linewidths=0.6, zorder=4
        )

    _, pval = mannwhitneyu(data[0], data[1], alternative="two-sided")
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"

    if ylim is not None:
        ax.set_ylim(*ylim)
        ymin, ymax = ylim
    else:
        all_v = np.concatenate(data)
        ymin, ymax = all_v.min(), all_v.max()
        yr = ymax - ymin
        ax.set_ylim(ymin - 0.12 * yr, ymax + 0.35 * yr)
        ymin, ymax = ax.get_ylim()

    yr = ymax - ymin
    bar_y = ymax - 0.13 * yr
    ax.plot([0, 1], [bar_y, bar_y], color="#222222", lw=0.9)
    ax.text(0.5, bar_y + 0.018 * yr, f"Mann-Whitney U: {sig} ($p$={pval:.3f})",
            ha="center", va="bottom", fontsize=10.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([DISPLAY_LABELS[lvl] for lvl in order], fontsize=12)
    ax.tick_params(axis="y", labelsize=10.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15.5, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#444444")


def draw_pcoa_panel(ax, df, group_col, order, perm_info, legend_loc="lower right"):
    for grp in order:
        sub = df[df[group_col].str.lower() == grp]
        col = COLORS[grp]
        mkr = MARKERS[grp]
        lbl = DISPLAY_LABELS[grp]

        ax.scatter(
            sub["PCo1"], sub["PCo2"],
            c=col, marker=mkr, s=135, zorder=4,
            edgecolors="#111111", linewidths=0.9, label=lbl
        )
        confidence_ellipse(
            sub["PCo1"].values, sub["PCo2"].values, ax,
            facecolor=col, alpha=0.12, edgecolor=col,
            linewidth=1.0, linestyle="-", zorder=2
        )
        cx, cy = sub["PCo1"].mean(), sub["PCo2"].mean()
        ax.plot(cx, cy, marker="+", color=col, ms=15, mew=2.4, zorder=5)

    for _, row in df.iterrows():
        ax.annotate(
            row["sample"], (row["PCo1"], row["PCo2"]),
            textcoords="offset points", xytext=(4, 3),
            fontsize=10.5, color="#111111", zorder=6
        )

    f = perm_info["f"]
    r2 = perm_info["r2"]
    p = perm_info["p"]
    sig = perm_info["sig"]
    mth = perm_info["method"]
    perm_txt = (
        f"PERMANOVA (adonis2)\n"
        f"$F$ = {f:.3f}, $R^2$ = {r2:.3f}, $p$ = {p:.4f} {sig}\n"
        f"{mth}"
    )
    ax.text(
        0.03, 0.97, perm_txt,
        transform=ax.transAxes, fontsize=11,
        va="top", ha="left",
        color="#1a3c5e" if p < 0.05 else "#4a4a4a",
        bbox=dict(
            boxstyle="round,pad=0.38", fc="white",
            ec="#9bb8d3" if p < 0.05 else "#cccccc",
            alpha=0.92
        )
    )

    ax.set_xlabel("PCoA1 (33.5%)", fontsize=15.5, fontweight="bold")
    ax.set_ylabel("PCoA2 (16.2%)", fontsize=15.5, fontweight="bold")
    ax.axhline(0, color="#cccccc", lw=0.8, ls="-", zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, ls="-", zorder=1)
    ax.legend(fontsize=12.5, framealpha=0.92, edgecolor="#888888", loc=legend_loc)
    ax.tick_params(axis="both", labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#444444")


def main():
    counts = pd.read_csv(RES / "adonis_counts.tsv", sep="\t", index_col=0)
    alpha_src = pd.read_csv(RES / "Alpha_diversity_12samples_PhuTho_SonLa.tsv", sep="\t")
    pcoa_df = pd.read_csv(RES / "PCoA_genus_BrayCurtis_12samples_coordinates.tsv", sep="\t")

    # Observed genus richness (number of genera with Bracken count > 0)
    richness_s = (counts > 0).sum(axis=1).rename("Observed_richness")
    alpha = alpha_src.set_index("sample").join(richness_s).reset_index()
    alpha["host"] = alpha["host"].str.lower().str.strip()
    alpha["condition"] = alpha["condition"].str.lower().str.strip()

    fig = plt.figure(figsize=(18.5, 13.8))
    gs_main = gridspec.GridSpec(2, 2, figure=fig, hspace=0.22, wspace=0.18)

    # Panel A: Observed genus richness (2 subpanels)
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 0], wspace=0.22)
    ax_a1 = fig.add_subplot(gs_a[0])
    ax_a2 = fig.add_subplot(gs_a[1])
    draw_alpha_subpanel(ax_a1, alpha, "Observed_richness", "host", ["citrus", "coffee"], ylabel="Observed genus richness", ylim=(1730, 2135))
    draw_alpha_subpanel(ax_a2, alpha, "Observed_richness", "condition", ["diseased", "healthy"], ylabel=None, ylim=(1730, 2135))

    # Panel B: Shannon (2 subpanels)
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 1], wspace=0.22)
    ax_b1 = fig.add_subplot(gs_b[0])
    ax_b2 = fig.add_subplot(gs_b[1])
    draw_alpha_subpanel(ax_b1, alpha, "Shannon", "host", ["citrus", "coffee"], ylabel="Shannon index", ylim=(3.73, 5.80))
    draw_alpha_subpanel(ax_b2, alpha, "Shannon", "condition", ["diseased", "healthy"], ylabel=None, ylim=(3.73, 5.80))

    # Panel C: PCoA Host
    ax_c = fig.add_subplot(gs_main[1, 0])
    draw_pcoa_panel(ax_c, pcoa_df, "host", ["citrus", "coffee"], PERMANOVA["host"], legend_loc="lower right")

    # Panel D: PCoA Condition
    ax_d = fig.add_subplot(gs_main[1, 1])
    draw_pcoa_panel(ax_d, pcoa_df, "condition", ["diseased", "healthy"], PERMANOVA["condition"], legend_loc="center right")

    # Panel labels A, B, C, D
    fig.text(0.015, 0.92, "A", fontsize=24, fontweight="bold", va="top", ha="left")
    fig.text(0.505, 0.92, "B", fontsize=24, fontweight="bold", va="top", ha="left")
    fig.text(0.015, 0.46, "C", fontsize=24, fontweight="bold", va="top", ha="left")
    fig.text(0.505, 0.46, "D", fontsize=24, fontweight="bold", va="top", ha="left")

    out_png = OUT_FIG / "Fig2.png"
    out_tif = OUT_FIG / "Fig2.tif"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    save_pace_tiff(out_png, out_tif)
    print(f"Saved: {out_png} and {out_tif}")



if __name__ == "__main__":
    main()
