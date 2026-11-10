"""
05_06_Figure5_Figure6_Species_Sankey_Bubble.py
==============================================
Reproduces Figure 5 and Figure 6 (2-Panel Species Sankey + MaAsLin2 Bubble Plot):
- Figure 5: Host/site-associated rhizosphere bacterial/archaeal species (Citrus vs. Coffee)
- Figure 6: Symptom-status-associated rhizosphere species (Symptomatic vs. Asymptomatic)
- Panel A: Two-column Sankey diagram linking enrichment group to top 30 differentially abundant species,
  with bacterial phylum encoded by a colored strip adjacent to each species node (Phylum_Color_Palette.tsv)
  and rectangular border outlines encoding the enrichment group.
- Panel B: MaAsLin2 multi-dimensional bubble plot showing log2 fold change, -log10(p-value), and prevalence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path
from PIL import Image

PKG_DIR = Path(__file__).resolve().parent.parent
RES = PKG_DIR / "Source_Data" / "results"
P1_DIR = PKG_DIR / "Source_Data" / "P1_inputs"
OUT_FIG_DIR = PKG_DIR / "Figures"
OUT_PANEL_DIR = OUT_FIG_DIR / "Panels"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PANEL_DIR.mkdir(parents=True, exist_ok=True)

PVAL_THRESH = 0.05
MAX_FEATURES = 30
FONT_DELTA = 5

C_CITRUS = "#D62728"
C_COFFEE = "#377EB8"
C_SYMPTOMATIC = "#FF7F0E"
C_ASYMPTOMATIC = "#90C978"

GROUP_COLORS = {
    "host": {"Citrus": C_CITRUS, "Coffee": C_COFFEE},
    "condition": {"Symptomatic": C_SYMPTOMATIC, "Asymptomatic": C_ASYMPTOMATIC},
}


plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.bf"] = "Arial:bold"


def format_species_label(name: str) -> str:
    """Italicize only the taxonomic genus/species words; keep 'sp.' and strain identifiers upright."""
    s = str(name).strip()
    if " sp. " in s or s.endswith(" sp."):
        parts = s.split(" sp.", 1)
        genus_part = parts[0].strip()
        strain_part = parts[1].strip()
        italic_genus = " ".join([f"$\\it{{{w}}}$" for w in genus_part.split()])
        if strain_part:
            return f"{italic_genus} sp. {strain_part}"
        return f"{italic_genus} sp."
    words = s.split()
    return " ".join([f"$\\it{{{w}}}$" for w in words])


def load_palette():
    palette_path = RES / "Phylum_Color_Palette.tsv"
    palette = {}
    if palette_path.exists():
        pal = pd.read_csv(palette_path, sep="\t")
        palette = dict(zip(pal["phylum"], pal["color"]))
    # Explicit unified mappings for archaeal/candidate phyla in Kraken2 species table
    palette.setdefault("Nitrososphaerota", "#B3DE69")
    palette.setdefault("Candidatus Saccharimonadota", "#6A3D9A")  # Patescibacteria / Saccharimonadota unified
    return palette


def load_taxonomy_species():
    tax = pd.read_csv(P1_DIR / "Taxonomy_long_with_species_12samples.tsv", sep="\t")
    if "excluded_host_eukaryote" in tax.columns:
        tax = tax[tax["excluded_host_eukaryote"].astype(str).str.lower() == "false"].copy()
    for col in ["host", "condition", "sample", "phylum", "genus", "species"]:
        if col in tax.columns:
            tax[col] = tax[col].astype(str).str.strip()
    tax["host"] = tax["host"].str.capitalize()
    tax["condition"] = tax["condition"].str.lower().replace({
        "diseased": "Symptomatic",
        "disease": "Symptomatic",
        "symptomatic": "Symptomatic",
        "healthy": "Asymptomatic",
        "asymptomatic": "Asymptomatic",
    })
    return tax


class Alluvial2WithPhylumStrip:
    """
    2-column Sankey diagram (Group -> Species) with a colored Phylum strip
    placed directly between each Species node box and the formatted Species name.
    """
    def __init__(self, fig_h=10.0, node_w=0.28, strip_w=0.22, pad=0.24, band_alpha=0.38):
        self.fig_h = fig_h
        self.node_w = node_w
        self.strip_w = strip_w
        self.pad = pad
        self.band_alpha = band_alpha
        self.min_node_h = 0.10
        self.node_sizes = {}
        self.node_pos = {}
        self.cur_out = {}
        self.cur_in = {}

    def add_flows(self, flows):
        for g, p, s, w in flows:
            self.node_sizes[(0, g)] = self.node_sizes.get((0, g), 0.0) + w
            self.node_sizes[(1, s)] = self.node_sizes.get((1, s), 0.0) + w

    def compute_layout(self):
        # Column 0 (Group): reserve bottom space [0.0, 3.85] for Phylum & Enrichment group legends
        col0_top = self.fig_h
        col0_bot_limit = 3.85
        col0_span = col0_top - col0_bot_limit
        items0 = sorted([(l, s) for (c, l), s in self.node_sizes.items() if c == 0], key=lambda i: -i[1])
        total0 = sum(s for _, s in items0)
        count0 = len(items0)
        pad0 = 0.35
        usable0 = col0_span - pad0 * max(count0 - 1, 0)
        min_h0 = 0.60
        extra0 = max(usable0 - min_h0 * count0, 0.0)
        y = col0_top
        for label, size in items0:
            frac = (size / total0) if total0 else 0.0
            height = min_h0 + frac * extra0
            y_bot = y - height
            self.node_pos[(0, label)] = (y_bot, height)
            self.cur_out[(0, label)] = y
            self.cur_in[(0, label)] = y
            y = y_bot - pad0

        # Column 1 (Species): full vertical height [0.0, fig_h]
        items1 = sorted([(l, s) for (c, l), s in self.node_sizes.items() if c == 1], key=lambda i: -i[1])
        total1 = sum(s for _, s in items1)
        count1 = len(items1)
        usable1 = self.fig_h - self.pad * max(count1 - 1, 0)
        min_h1 = min(self.min_node_h, usable1 / max(count1, 1) * 0.55)
        extra1 = max(usable1 - min_h1 * count1, 0.0)
        y = self.fig_h
        for label, size in items1:
            frac = (size / total1) if total1 else 0.0
            height = min_h1 + frac * extra1
            y_bot = y - height
            self.node_pos[(1, label)] = (y_bot, height)
            self.cur_out[(1, label)] = y
            self.cur_in[(1, label)] = y
            y = y_bot - self.pad

    def _draw_band(self, ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color):
        t = np.linspace(0, 1, 120)
        xm = (x0 + x1) / 2.0
        xt = (1 - t)**3 * x0 + 3*(1 - t)**2 * t * xm + 3*(1 - t)*t**2 * xm + t**3 * x1
        yt = (1 - t)**3 * y0_top + 3*(1 - t)**2 * t * y0_top + 3*(1 - t)*t**2 * y1_top + t**3 * y1_top
        xb = (1 - t)**3 * x1 + 3*(1 - t)**2 * t * xm + 3*(1 - t)*t**2 * xm + t**3 * x0
        yb = (1 - t)**3 * y1_bot + 3*(1 - t)**2 * t * y1_bot + 3*(1 - t)*t**2 * y0_bot + t**3 * y0_bot
        verts = np.column_stack([np.concatenate([xt, xb]), np.concatenate([yt, yb])])
        ax.add_patch(mpatches.Polygon(
            verts, closed=True, facecolor=color, edgecolor="none",
            alpha=self.band_alpha, zorder=2
        ))

    def draw(self, ax, col_xs, flows, palette, group_colors):
        self.add_flows(flows)
        self.compute_layout()

        # Map species -> enriched group and species -> phylum
        sp_to_grp = {}
        sp_to_phy = {}
        g_s = {}
        for g, p, s, w in flows:
            g_s[(g, s)] = g_s.get((g, s), 0.0) + w
            if s not in sp_to_grp:
                sp_to_grp[s] = g
            if s not in sp_to_phy:
                sp_to_phy[s] = p

        def y_center(c, l):
            if (c, l) not in self.node_pos:
                return 0.0
            return self.node_pos[(c, l)][0] + self.node_pos[(c, l)][1] / 2.0

        link_list = [(g, s, w) for (g, s), w in g_s.items()]
        sorted_links = sorted(link_list, key=lambda i: (-y_center(0, i[0]), -y_center(1, i[1])))
        x_from = col_xs[0] + self.node_w / 2 - 0.02
        x_to = col_xs[1] - self.node_w / 2 + 0.02

        for src, tgt, weight in sorted_links:
            sk, tk = (0, src), (1, tgt)
            if sk not in self.node_pos or tk not in self.node_pos:
                continue
            out_h = (weight / self.node_sizes.get(sk, 1.0)) * self.node_pos[sk][1]
            in_h = (weight / self.node_sizes.get(tk, 1.0)) * self.node_pos[tk][1]
            y0_top, y1_top = self.cur_out[sk], self.cur_in[tk]
            y0_bot, y1_bot = y0_top - out_h, y1_top - in_h
            self.cur_out[sk], self.cur_in[tk] = y0_bot, y1_bot
            if out_h > 0 and in_h > 0:
                band_col = palette.get(sp_to_phy.get(tgt, "Other"), "#BDBDBD")
                self._draw_band(ax, x_from, y0_bot, y0_top, x_to, y1_bot, y1_top, band_col)

        # Draw nodes & phylum strips
        x_strip_left = col_xs[1] + self.node_w / 2 + 0.08
        for (col, label), (y_bot, height) in self.node_pos.items():
            x = col_xs[col]
            # +1 unit species font size as requested
            fs_sp = (10.5 if len(str(label)) < 20 else 9.8) + FONT_DELTA + 1.5
            fs_grp = 9.5 + FONT_DELTA + 2.5
            if col == 0:
                fc = group_colors.get(label, "#DDDDDD")
                ec = fc
                lw = 1.6
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x - self.node_w / 2, y_bot), self.node_w, height,
                    boxstyle="square,pad=0", facecolor=fc, edgecolor=ec, linewidth=lw, zorder=5
                ))
                ax.text(
                    x + self.node_w / 2 + 0.12, y_bot + height / 2, str(label),
                    ha="left", va="center", fontsize=fs_grp, fontweight="bold",
                    fontfamily="Arial", zorder=6
                )
            else:
                sg = sp_to_grp.get(label, "")
                phy = sp_to_phy.get(label, "Other")
                fc = "#F4F4F4"
                ec = group_colors.get(sg, "#777777")
                lw = 2.0
                # Species box (outlined by enriched host/condition group)
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x - self.node_w / 2, y_bot), self.node_w, height,
                    boxstyle="square,pad=0", facecolor=fc, edgecolor=ec, linewidth=lw, zorder=5
                ))
                # Adjacent Phylum color strip
                phy_col = palette.get(phy, "#BDBDBD")
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x_strip_left, y_bot), self.strip_w, height,
                    boxstyle="square,pad=0", facecolor=phy_col, edgecolor="#333333", linewidth=0.7, zorder=5
                ))
                # Species label to the right of the Phylum strip (only genus/species italicized, sp. and strain upright)
                ax.text(
                    x_strip_left + self.strip_w + 0.12, y_bot + height / 2, format_species_label(label),
                    ha="left", va="center", fontsize=fs_sp, fontfamily="Arial",
                    style="normal", zorder=6
                )

        # Column headers
        ax.text(col_xs[0], self.fig_h + 0.45, "Group", ha="center", va="bottom",
                fontsize=12 + FONT_DELTA + 2, fontweight="bold", fontfamily="Arial")
        ax.text(col_xs[1] + 0.15, self.fig_h + 0.45, "Species", ha="center", va="bottom",
                fontsize=12 + FONT_DELTA + 2, fontweight="bold", fontfamily="Arial")

        # Add Phylum legend & Enrichment group legend flush to the left margin under the Group bar
        present_phyla = []
        for _, p, _, _ in flows:
            if p not in present_phyla:
                present_phyla.append(p)

        phylum_handles = [
            mpatches.Patch(facecolor=palette.get(p, "#BDBDBD"), edgecolor="#333333", linewidth=0.8, label=p)
            for p in present_phyla
        ]
        outline_handles = [
            mpatches.Patch(facecolor="#F4F4F4", edgecolor=c, linewidth=2.0, label=f"{g}-enriched")
            for g, c in group_colors.items()
        ]

        is_cond = "Symptomatic" in group_colors
        leg_x = -0.015 if is_cond else -0.005
        leg_phy_y = -0.075 if is_cond else -0.025
        leg_out_y = (0.072 if is_cond else (0.102 if len(present_phyla) <= 4 else 0.122))

        leg_phy = ax.legend(
            handles=phylum_handles,
            loc="lower left",
            bbox_to_anchor=(leg_x, leg_phy_y),
            title="Phylum",
            fontsize=11.5,
            title_fontsize=12.5,
            framealpha=0.94,
            edgecolor="#CCCCCC",
            borderpad=0.45,
            labelspacing=0.32,
            prop={"family": "Arial", "style": "normal", "size": 11.5},
        )
        leg_phy.get_title().set_fontweight("bold")
        leg_phy.get_title().set_fontfamily("Arial")
        ax.add_artist(leg_phy)

        leg_out = ax.legend(
            handles=outline_handles,
            loc="lower left",
            bbox_to_anchor=(leg_x, leg_out_y),
            title="Enrichment group",
            fontsize=11.5,
            title_fontsize=12.5,
            framealpha=0.94,
            edgecolor="#CCCCCC",
            borderpad=0.45,
            labelspacing=0.32,
            prop={"family": "Arial", "style": "normal", "size": 11.5},
        )
        leg_out.get_title().set_fontweight("bold")
        leg_out.get_title().set_fontfamily("Arial")

        ax.set_xlim(col_xs[0] - 0.15, col_xs[1] + 2.55)
        ax.set_ylim(-0.4, self.fig_h + 1.05)
        ax.axis("off")


def draw_bubble_plot(ax, res_df, mode):
    sub = res_df.copy()
    if sub.empty:
        return
    sub["log_p"] = -np.log10(np.clip(sub["pval"], 1e-10, 1))
    sub["dot_col"] = sub["coef"].apply(
        lambda x: C_CITRUS if (mode == "host" and x > 0)
        else (C_COFFEE if mode == "host" else (C_SYMPTOMATIC if x > 0 else C_ASYMPTOMATIC))
    )

    sub = sub.sort_values(by="coef", ascending=True)
    ax.axvline(0, ls="--", color="#888888", lw=0.7)

    min_lp = sub["log_p"].min()
    max_lp = sub["log_p"].max()
    lp_range = max_lp - min_lp + 1e-9

    sizes = (sub["log_p"] - min_lp) / lp_range
    sizes = sizes * 400 + 80

    ax.scatter(
        sub["coef"], range(len(sub)), s=sizes, c=sub["dot_col"],
        edgecolor="black", alpha=0.85, linewidth=0.5, zorder=3
    )

    span = sub["coef"].max() - sub["coef"].min()
    ax.set_xlim(sub["coef"].min() - 0.13 * span, sub["coef"].max() + 0.13 * span)

    for i, (_, row) in enumerate(sub.iterrows()):
        star = "***" if row["pval"] < 0.001 else ("**" if row["pval"] < 0.01 else "*")
        norm_r = ((row["log_p"] - min_lp) / lp_range) ** 0.5
        offset = (0.018 + 0.020 * norm_r) * span
        x_pos = row["coef"] + offset if row["coef"] >= 0 else row["coef"] - offset
        ha = "left" if row["coef"] >= 0 else "right"
        ax.text(x_pos, i, star, va="center", ha=ha, fontsize=10, fontweight="bold", color="#333333")

    ax.set_yticks(range(len(sub)))
    # +1 unit species font size and only species/genus words italicized (sp. and strain upright)
    ax.set_yticklabels([format_species_label(s) for s in sub["species"]], fontstyle="normal", fontsize=10 + FONT_DELTA + 1)
    ax.tick_params(axis="x", labelsize=10 + FONT_DELTA + 2)
    ax.set_xlabel("MaAsLin2 coefficient", fontsize=11 + FONT_DELTA + 2)

    ax.text(
        1.0, 1.02, "Significance: *$\\it{p}$ < 0.05; **$\\it{p}$ < 0.01; ***$\\it{p}$ < 0.001",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10 + FONT_DELTA
    )
    ax.grid(True, axis="x", ls=":", color="#dddddd")

    if mode == "host":
        colors_lbls = [("Citrus-enriched", C_CITRUS), ("Coffee-enriched", C_COFFEE)]
    else:
        colors_lbls = [("Symptomatic-enriched", C_SYMPTOMATIC), ("Asymptomatic-enriched", C_ASYMPTOMATIC)]

    color_handles = [
        mlines.Line2D([], [], color=c, marker="o", linestyle="None", markersize=10, label=lbl)
        for lbl, c in colors_lbls
    ]

    ticks = [min_lp, (min_lp + max_lp) / 2, max_lp]
    size_handles = []
    for lp in ticks:
        area = ((lp - min_lp) / lp_range) * 400 + 80
        size_handles.append(
            ax.scatter([], [], s=area, color="#555555", alpha=0.6,
                       label=f"$-\\log_{{10}}(\\it{{p}}\\text{{-value}}) \\sim {lp:.1f}$")
        )

    leg2 = ax.legend(
        handles=size_handles, loc="lower right",
        title="Bubble size",
        fontsize=11 + FONT_DELTA - 2, title_fontsize=12 + FONT_DELTA - 2,
        framealpha=0.9, scatterpoints=1,
        labelspacing=1.8, borderpad=1.2
    )
    ax.add_artist(leg2)

    ax.legend(
        handles=color_handles, loc="lower right",
        bbox_to_anchor=(1.0, 0.22),
        title="Enrichment group",
        fontsize=9 + FONT_DELTA - 2, title_fontsize=10 + FONT_DELTA - 2,
        framealpha=0.9
    )


def save_fig_formats(fig, base_name, legacy_name=None, max_w=4500, max_h=5250, dpi=600):
    Image.MAX_IMAGE_PIXELS = None
    png_path = OUT_FIG_DIR / f"{base_name}.png"
    tif_path = OUT_FIG_DIR / f"{base_name}.tif"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    target_scale = min(max_w / w, max_h / h, 1.0)
    if target_scale < 1.0:
        img = img.resize((int(round(w * target_scale)), int(round(h * target_scale))), Image.Resampling.LANCZOS)
        img.save(png_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    img.save(tif_path, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
    print(f"Saved: {png_path.name}, {tif_path.name} ({img.size[0]}x{img.size[1]} px)")




def main():
    tax = load_taxonomy_species()
    palette = load_palette()

    lm_host = pd.read_csv(P1_DIR / "species_lm_host.tsv", sep="\t")
    lm_cond = pd.read_csv(P1_DIR / "species_lm_cond.tsv", sep="\t")

    def process_mode(lm, mode):
        sig = lm[lm["pval"] < PVAL_THRESH].copy()
        sig = sig.reindex(sig["coef"].abs().sort_values(ascending=False).index).head(MAX_FEATURES)

        species_rows = tax[tax["rank"] == "species"]
        s2p = species_rows.groupby("species")["phylum"].first().to_dict()
        weight_df = (
            species_rows.groupby(["sample", "species"], as_index=False)["percent"]
            .sum()
            .merge(tax[["sample", "host", "condition"]].drop_duplicates(), on="sample", how="left")
        )

        log_p = -np.log10(np.clip(sig["pval"], 1e-10, 1))
        lp_min, lp_max = log_p.min(), log_p.max()
        bub_norm = (log_p - lp_min) / (lp_max - lp_min + 1e-9)
        max_coef = sig["coef"].abs().max() + 1e-9

        pos_grp, neg_grp = ("Citrus", "Coffee") if mode == "host" else ("Symptomatic", "Asymptomatic")
        flows = []
        for idx, row in sig.iterrows():
            sp = row["species"]
            grp = pos_grp if row["coef"] > 0 else neg_grp
            if mode == "host":
                w = weight_df[(weight_df[mode] == grp) & (weight_df["species"] == sp)]["percent"].mean()
                if pd.isna(w) or w <= 0:
                    w = max(abs(row["coef"]) * 0.002, 0.002)
            else:
                # Proportional to Panel B bubble size (-log10 p) and effect size (|coef|)
                bn = bub_norm.loc[idx]
                w = ((bn + 0.08) * ((abs(row["coef"]) / max_coef) ** 1.2)) ** 2.4
            flows.append((grp, s2p.get(sp, "Other"), sp, w))

        return flows, sig

    h_flows, h_sig = process_mode(lm_host, "host")
    c_flows, c_sig = process_mode(lm_cond, "condition")

    # ---------------- Figure 5: Host Sankey + Bubble ----------------
    fig = plt.figure(figsize=(21.5, 12.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1])
    ax_sank = fig.add_subplot(gs[0])
    ax_bub = fig.add_subplot(gs[1])

    Alluvial2WithPhylumStrip(fig_h=10).draw(ax_sank, [0.45, 2.80], h_flows, palette, GROUP_COLORS["host"])
    draw_bubble_plot(ax_bub, h_sig, "host")

    ax_sank.text(-0.01, 1.03, "A", transform=ax_sank.transAxes, fontsize=28, fontweight="bold", fontfamily="Arial", va="top", ha="left")
    ax_bub.text(-0.22, 1.03, "B", transform=ax_bub.transAxes, fontsize=28, fontweight="bold", fontfamily="Arial", va="top", ha="left")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.80)
    save_fig_formats(fig, "Fig5", "Figure 5")
    plt.close(fig)

    # ---------------- Figure 6: Condition Sankey + Bubble ----------------
    fig = plt.figure(figsize=(21.5, 12.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1])
    ax_sank = fig.add_subplot(gs[0])
    ax_bub = fig.add_subplot(gs[1])

    Alluvial2WithPhylumStrip(fig_h=10).draw(ax_sank, [0.45, 2.80], c_flows, palette, GROUP_COLORS["condition"])
    draw_bubble_plot(ax_bub, c_sig, "condition")

    ax_sank.text(-0.01, 1.03, "A", transform=ax_sank.transAxes, fontsize=28, fontweight="bold", fontfamily="Arial", va="top", ha="left")
    ax_bub.text(-0.22, 1.03, "B", transform=ax_bub.transAxes, fontsize=28, fontweight="bold", fontfamily="Arial", va="top", ha="left")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.65)
    save_fig_formats(fig, "Fig6", "Figure 6")
    plt.close(fig)


if __name__ == "__main__":
    main()
