"""
03_Figure3_MaAsLin2_KEGG_KO.py
==============================
Reproduces Figure 3: Top 25 differentially abundant KEGG Orthologs (KOs) identified by MaAsLin2
  - Panel A: Host comparison (Citrus-enriched vs. Coffee-enriched)
  - Panel B: Plant health status comparison (Symptomatic-enriched vs. Asymptomatic-enriched)

Outputs:
  - Figures/Fig3.png (600 DPI)
  - Figures/Fig3.tif (600 DPI, LZW-compressed RGB TIFF, PACE-compliant)
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PKG_DIR = Path(__file__).resolve().parent.parent
OUT_FIG = PKG_DIR / "Figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)


def save_pace_tiff(png_path: Path, tif_path: Path, max_w: int = 4500, max_h: int = 5250, dpi: int = 600):
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    target_scale = min(max_w / w, max_h / h, 1.0)
    if target_scale < 1.0:
        img = img.resize((int(round(w * target_scale)), int(round(h * target_scale))), Image.LANCZOS)
        img.save(png_path, format="PNG", dpi=(dpi, dpi), optimize=True)
    img.save(tif_path, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))



def get_stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return ""


def create_panel(ax, df: pd.DataFrame, metadata: str, group_names: list[str], colors: list[str], max_items: int = 25):
    df_meta = df[df["metadata"] == metadata].copy()
    df_meta["abs_coef"] = df_meta["coef"].abs()
    top_features = df_meta.sort_values("abs_coef", ascending=False).head(max_items)
    top_features = top_features.sort_values("coef", ascending=True)

    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features["coef"], align="center", edgecolor="black", linewidth=0.5)

    for i, (_, row) in enumerate(top_features.iterrows()):
        if row["coef"] > 0:
            bars[i].set_color(colors[0])
            bars[i].set_edgecolor("black")
            ax.text(row["coef"] + 0.1, i, get_stars(row["pval"]), va="center")
        else:
            bars[i].set_color(colors[1])
            bars[i].set_edgecolor("black")
            ax.text(row["coef"] - 0.1, i, get_stars(row["pval"]), va="center", ha="right")

    labels = []
    for _, row in top_features.iterrows():
        label = str(row["Function description"])
        if len(label) > 60:
            label = label[:57] + "..."
        labels.append(f"{row['feature']}: {label}")

    min_coef = float(top_features["coef"].min())
    max_coef = float(top_features["coef"].max())
    span = max_coef - min_coef
    ax.set_xlim(min_coef - 0.18 * span, max_coef + 0.24 * span)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("MaAsLin2 Coefficient", fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    legend_elements = [
        Patch(facecolor=colors[0], edgecolor="black", label=group_names[0]),
        Patch(facecolor=colors[1], edgecolor="black", label=group_names[1]),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.95)


def main():
    df = pd.read_csv(PKG_DIR / "Source_Data/P1_inputs/MaAsLin2_KEGG_pval005_annotated.tsv", sep="\t")
    df = df.dropna(subset=["pval"])
    df["pval"] = df["pval"].astype(float)
    df["coef"] = df["coef"].astype(float)
    df = df[df["pval"] < 0.05]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(23, 12))

    create_panel(ax1, df, "host", ["Citrus-enriched", "Coffee-enriched"], ["#e41a1c", "#377eb8"])
    ax1.set_title("A", loc="left", fontsize=20, fontweight="bold")

    create_panel(ax2, df, "condition", ["Symptomatic-enriched", "Asymptomatic-enriched"], ["#ff7f00", "#4daf4a"])
    ax2.set_title("B", loc="left", fontsize=20, fontweight="bold")

    plt.suptitle("Significance: *$p$ < 0.05; **$p$ < 0.01; ***$p$ < 0.001", fontsize=15, y=0.92)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])

    out_png = OUT_FIG / "Fig3.png"
    out_tif = OUT_FIG / "Fig3.tif"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    save_pace_tiff(out_png, out_tif)
    print(f"Saved: {out_png} and {out_tif}")



if __name__ == "__main__":
    main()
