"""
04_Figure4_RF_KEGG_KO.py
========================
Reproduces Figure 4: Random Forest classification (LOOCV ROC curves) and top 25 discriminating KEGG Orthologs (KOs)
  - Panel A: Host classification ROC curve (Citrus vs. Coffee)
  - Panel B: Top 25 discriminating KOs by Host (Mean decrease Gini)
  - Panel C: Plant health status classification ROC curve (Symptomatic vs. Asymptomatic)
  - Panel D: Top 25 discriminating KOs by Plant Health Status (Symptomatic vs. Asymptomatic; Mean decrease Gini)

Outputs:
  - Figures/Fig4.png (600 DPI)
  - Figures/Fig4.tif (600 DPI, LZW-compressed RGB TIFF, PACE-compliant)
"""
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rf_validation_and_combined_loocv import (
    load_metadata,
    load_ko_map,
    load_kegg_matrix,
    loocv_auc,
    compute_direction_table,
    make_rf,
    COL_HOST, COL_STATUS,
    HOST_POS, HOST_NEG,
    STATUS_POS, STATUS_NEG,
    RANDOM_SEED
)

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



def get_loocv_importance(X, y, n_estimators=700, seed=RANDOM_SEED):
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    loo = LeaveOneOut()
    importances = []

    for train_index, _ in loo.split(X):
        X_train = X.iloc[train_index]
        y_train = y_enc[train_index]
        rf = make_rf(seed, n_estimators=n_estimators)
        rf.fit(X_train, y_train)
        importances.append(rf.feature_importances_)

    importances = np.array(importances)
    return pd.DataFrame({
        "feature": X.columns,
        "importance": np.mean(importances, axis=0),
        "importance_std": np.std(importances, axis=0),
    })


def plot_roc_curve(ax, fpr, tpr, auc_val, title, color, ci_text=""):
    label = f"mean AUC = {auc_val:.3f}"
    if ci_text:
        label += f" ({ci_text})"
    ax.plot(fpr, tpr, color=color, lw=2, label=label)
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.set_title(title, fontsize=14, pad=10, loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=12)


def plot_signed_importance(
    ax, table: pd.DataFrame, positive_label: str, negative_label: str,
    positive_color: str, negative_color: str, title: str,
    pos_display: str = None, neg_display: str = None
):
    pos_display = pos_display or positive_label
    neg_display = neg_display or negative_label

    top = table.sort_values("signed_importance", ascending=True).copy().reset_index(drop=True)
    colors = [positive_color if grp == positive_label else negative_color for grp in top["favored_group"]]

    y_pos = np.arange(len(top))
    ax.barh(
        y_pos, top["signed_importance"], xerr=top["importance_std"],
        color=colors, edgecolor="#2B2B2B", linewidth=0.6,
        error_kw={"ecolor": "gray", "capsize": 2, "alpha": 0.7}
    )

    for i, row in top.iterrows():
        val = row["signed_importance"]
        desc = row["feature_label"] if "feature_label" in row else row["feature"]
        desc_short = desc.split("\n")[0]
        if len(desc_short) > 55:
            desc_short = desc_short[:52] + "..."

        if val > 0:
            ax.text(-0.0001, i, desc_short, ha="right", va="center", fontsize=10, color="#2B2B2B")
        else:
            ax.text(0.0001, i, desc_short, ha="left", va="center", fontsize=10, color="#2B2B2B")

    ax.set_xlabel("Mean decrease Gini", fontsize=12)
    ax.tick_params(axis="x", which="major", labelsize=11)
    ax.set_title(title, fontsize=14, pad=10, loc="left", fontweight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_val = np.max(np.abs(top["signed_importance"]) + top["importance_std"])
    lim = float(max_val) * 1.1 if len(top) > 0 else 1.0
    ax.set_xlim(-lim, lim)

    p_patch = mpatches.Patch(color=positive_color, label=pos_display)
    n_patch = mpatches.Patch(color=negative_color, label=neg_display)
    ax.legend(
        handles=[p_patch, n_patch], loc="upper left", bbox_to_anchor=(1.02, 1),
        frameon=False, fontsize=13, handlelength=1.5, handleheight=1.5
    )


def main():
    meta = load_metadata()
    ko_map = load_ko_map()
    Xkegg = load_kegg_matrix()

    print("Running LOOCV for Host...")
    host_res = loocv_auc(Xkegg, meta["host"], n_estimators=700)
    host_cv_imp = get_loocv_importance(Xkegg, meta["host"], n_estimators=700)
    host_dir = compute_direction_table(Xkegg, host_res["y"], host_cv_imp, HOST_POS, HOST_NEG, ko_map)
    host_top = host_dir.assign(abs_imp=lambda x: x["signed_importance"].abs()).sort_values("abs_imp", ascending=False).head(25).copy()

    print("Running LOOCV for Plant Health Status...")
    status_res = loocv_auc(Xkegg, meta["condition"], n_estimators=700)
    status_cv_imp = get_loocv_importance(Xkegg, meta["condition"], n_estimators=700)
    status_dir = compute_direction_table(Xkegg, status_res["y"], status_cv_imp, STATUS_POS, STATUS_NEG, ko_map)
    status_top = status_dir.assign(abs_imp=lambda x: x["signed_importance"].abs()).sort_values("abs_imp", ascending=False).head(25).copy()
    host_dir.to_csv(PKG_DIR / "Source_Data/MaAsLin2/RF_Host_KEGG_signed_importance.tsv", sep="\t", index=False)
    status_dir.to_csv(PKG_DIR / "Source_Data/MaAsLin2/RF_Status_KEGG_signed_importance.tsv", sep="\t", index=False)

    fig = plt.figure(figsize=(17, 13.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], wspace=0.25, hspace=0.15)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_roc_curve(ax_a, host_res["fpr"], host_res["tpr"], host_res["auc"], "A", "#2ca02c", "95% CI: 1.000-1.000")
    plot_signed_importance(ax_b, host_top, HOST_POS, HOST_NEG, COL_HOST[HOST_POS], COL_HOST[HOST_NEG], "B")

    plot_roc_curve(ax_c, status_res["fpr"], status_res["tpr"], status_res["auc"], "C", "#1f77b4", "95% CI: 0.505-0.995")
    plot_signed_importance(
        ax_d, status_top, STATUS_POS, STATUS_NEG,
        COL_STATUS[STATUS_POS], COL_STATUS[STATUS_NEG], "D",
        pos_display="Asymptomatic", neg_display="Symptomatic"
    )

    out_png = OUT_FIG / "Fig4.png"
    out_tif = OUT_FIG / "Fig4.tif"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    save_pace_tiff(out_png, out_tif)
    print(f"Saved: {out_png} and {out_tif}")



if __name__ == "__main__":
    main()
