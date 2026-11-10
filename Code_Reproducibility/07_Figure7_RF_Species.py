"""
Reproducibility Script: Figure 7 (Random Forest Species-Level Classification & Top 25 Biomarkers)
- Panel A: Host classification ROC curve (Citrus vs. Coffee)
- Panel B: Top 25 host-discriminating species (Mean decrease Gini)
- Panel C: Phytosanitary status classification ROC curve (Symptomatic vs. Asymptomatic)
- Panel D: Top 25 status-discriminating species (Mean decrease Gini)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from PIL import Image

PKG_DIR = Path(__file__).resolve().parent.parent
P1_DIR = PKG_DIR / "Source_Data" / "P1_inputs"
OUT_FIG_DIR = PKG_DIR / "Figures"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

HOST_POS, HOST_NEG = "Citrus", "Coffee"
STATUS_POS, STATUS_NEG = "Symptomatic", "Asymptomatic"
HOST_COLORS = {HOST_POS: "#D62728", HOST_NEG: "#377EB8"}
STATUS_COLORS = {STATUS_POS: "#FF7F0E", STATUS_NEG: "#90C978"}
TOP_FEATURES = 25


def load_data():
    df = pd.read_csv(P1_DIR / "Taxonomy_long_with_species_12samples.tsv", sep="\t")
    species = df[df["rank"] == "species"]
    matrix = species.pivot(index="species", columns="sample", values="clade_reads").fillna(0)
    X = matrix.div(matrix.sum(axis=0).replace(0, np.nan), axis=1).fillna(0).T

    m = species[["sample", "host", "condition"]].drop_duplicates().set_index("sample")
    m["host"] = m["host"].str.capitalize()
    m["condition"] = m["condition"].str.lower().replace({
        "diseased": "Symptomatic",
        "disease": "Symptomatic",
        "symptomatic": "Symptomatic",
        "healthy": "Asymptomatic",
        "asymptomatic": "Asymptomatic",
    })

    common = X.index.intersection(m.index)
    return X.loc[common], m.loc[common]


def evaluate_rf(X, y_raw, seed=42):
    le = LabelEncoder()
    # Binary label encoding: Symptomatic = 0, Asymptomatic = 1
    if set(y_raw.unique()) == {"Symptomatic", "Asymptomatic"}:
        y = np.where(y_raw == "Symptomatic", 0, 1)
    else:
        y = le.fit_transform(y_raw)
    min_class_n = int(np.min(np.bincount(y)))
    n_splits = max(2, min(5, min_class_n))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rf = RandomForestClassifier(
        n_estimators=700, max_features="sqrt", class_weight="balanced",
        random_state=seed, n_jobs=1
    )

    mean_fpr = np.linspace(0, 1, 120)
    tprs, aucs = [], []
    for tr, te in cv.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        rf.fit(Xtr, ytr)
        prob = rf.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, prob)
        aucs.append(auc(fpr, tpr))
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    # Also compute exact Leave-One-Out Cross-Validation (LOOCV) AUC
    loo = LeaveOneOut()
    loo_probs = np.zeros(len(y), dtype=float)
    for tr, te in loo.split(X):
        rf.fit(X.iloc[tr], y[tr])
        loo_probs[te[0]] = rf.predict_proba(X.iloc[te])[:, 1][0]
    loocv_auc_val = float(roc_auc_score(y, loo_probs))

    rf.fit(X, y)
    tree_imps = np.array([est.feature_importances_ for est in rf.estimators_])
    imp_mean = tree_imps.mean(axis=0)
    imp_sem = tree_imps.std(axis=0) / np.sqrt(tree_imps.shape[0])

    mean_auc = float(np.mean(aucs))
    sd_auc = float(np.std(aucs))
    ci_lo = max(0.0, mean_auc - 1.96 * sd_auc / np.sqrt(n_splits))
    ci_hi = min(1.0, mean_auc + 1.96 * sd_auc / np.sqrt(n_splits))

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": np.mean(tprs, axis=0),
        "mean_auc": mean_auc,
        "loocv_auc": loocv_auc_val,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "importance": imp_mean,
        "importance_std": imp_sem,
    }



def get_directions(X, y, pos_lbl, neg_lbl, imp_mean, imp_std):
    pos_samples = y[y == pos_lbl].index
    neg_samples = y[y == neg_lbl].index
    mean_pos = X.loc[pos_samples].mean(axis=0)
    mean_neg = X.loc[neg_samples].mean(axis=0)

    df = pd.DataFrame({"feature": X.columns, "importance": imp_mean, "std": imp_std})
    df["favored"] = np.where(mean_pos.values >= mean_neg.values, pos_lbl, neg_lbl)
    df["signed_imp"] = np.where(df["favored"] == pos_lbl, df["importance"], -df["importance"])
    return df


def plot_roc(ax, res, title):
    ax.plot(res["mean_fpr"], res["mean_tpr"], color="#111111", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    txt = f"AUC = {res['mean_auc']:.3f}\n95% CI [{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]"
    ax.text(
        0.55, 0.14, txt, transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9)
    )


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


def plot_bars(ax, df, pos_lbl, neg_lbl, c_pos, c_neg, title):
    top = df.sort_values("importance", ascending=False).head(TOP_FEATURES).copy()
    neg_block = top[top["favored"] == neg_lbl].sort_values("importance", ascending=False)
    pos_block = top[top["favored"] == pos_lbl].sort_values("importance", ascending=False)
    top = pd.concat([neg_block, pos_block], ignore_index=True)

    colors = [c_pos if f == pos_lbl else c_neg for f in top["favored"]]
    ax.barh(
        range(len(top)), top["signed_imp"], xerr=top["std"],
        color=colors, edgecolor="#2B2B2B", linewidth=0.6,
        error_kw=dict(ecolor="#333333", elinewidth=0.9, capsize=3, capthick=0.9)
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([format_species_label(s) for s in top["feature"]], fontstyle="normal", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease Gini", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    lim = float(top["signed_imp"].abs().max()) + float(top["std"].max())
    ax.set_xlim(-lim * 1.25, lim * 1.25)
    if len(neg_block) and len(pos_block):
        ax.axhline(len(neg_block) - 0.5, color="#BDBDBD", lw=0.8)

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=c_pos, edgecolor="black", label=f"{pos_lbl}-enriched"),
            mpatches.Patch(facecolor=c_neg, edgecolor="black", label=f"{neg_lbl}-enriched"),
        ],
        loc="lower right", fontsize=10, framealpha=0.9
    )


def main():
    X, m = load_data()
    res_h = evaluate_rf(X, m["host"])
    dir_h = get_directions(X, m["host"], HOST_POS, HOST_NEG, res_h["importance"], res_h["importance_std"])

    res_s = evaluate_rf(X, m["condition"])
    dir_s = get_directions(X, m["condition"], STATUS_POS, STATUS_NEG, res_s["importance"], res_s["importance_std"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw={"width_ratios": [1, 1.2], "height_ratios": [1, 1]})
    plot_roc(axes[0, 0], res_h, "(A) Host classification (ROC)")
    plot_bars(axes[0, 1], dir_h, HOST_POS, HOST_NEG, HOST_COLORS[HOST_POS], HOST_COLORS[HOST_NEG], f"(B) Top {TOP_FEATURES} Discriminating Species (Host)")
    plot_roc(axes[1, 0], res_s, "(C) Health status classification (ROC)")
    plot_bars(axes[1, 1], dir_s, STATUS_POS, STATUS_NEG, STATUS_COLORS[STATUS_POS], STATUS_COLORS[STATUS_NEG], f"(D) Top {TOP_FEATURES} Discriminating Species (Status)")

    plt.tight_layout()
    png_path = OUT_FIG_DIR / "Fig7.png"
    tif_path = OUT_FIG_DIR / "Fig7.tif"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

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
