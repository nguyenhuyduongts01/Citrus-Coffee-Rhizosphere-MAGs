# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy", "matplotlib", "scikit-learn"]
# ///
from pathlib import Path
import re
import textwrap
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

PKG_DIR = Path(__file__).resolve().parent.parent
RES = PKG_DIR / "Source_Data" / "results"
MAASLIN_DIR = PKG_DIR / "Source_Data" / "MaAsLin2"
FIG = PKG_DIR / "Figures"
OUT = PKG_DIR / "Source_Data" / "results"
PUB = PKG_DIR / "Figures"
PUB.mkdir(parents=True, exist_ok=True)

HOST_POS = "Citrus"
HOST_NEG = "Coffee"
STATUS_POS = "Healthy"       # Asymptomatic group
STATUS_NEG = "Disease"       # Symptomatic group
STATUS_DISPLAY = {"Healthy": "Asymptomatic", "Disease": "Symptomatic"}
COL_HOST = {HOST_POS: "#E41A1C", HOST_NEG: "#377EB8"}
COL_STATUS = {STATUS_POS: "#90C978", STATUS_NEG: "#FF7F0E"}
RANDOM_SEED = 42
MAIN_TREES = 700
PERMUTATION_TREES = 700
N_PERMUTATIONS = 1000
TOP_K_KEGG = 15
TOP_K_GENUS = 15


warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with",
)


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(RES / "Metadata_12samples_MaAsLin2.tsv", sep="\t")
    meta = meta[["sample", "host", "condition"]].drop_duplicates("sample").set_index("sample")
    meta["host"] = meta["host"].astype(str).str.strip().str.title()
    meta["condition"] = meta["condition"].astype(str).str.strip().str.title()
    return meta.loc[[f"S{i}" for i in range(1, 13) if f"S{i}" in meta.index]].copy()


def load_ko_map() -> dict[str, str]:
    candidates = [RES / "ko_description_map.tsv", RES / "KEGG_KO_mapping.tsv"]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            cols = {c.lower(): c for c in df.columns}
            ko_col = cols.get("ko_id") or cols.get("ko") or cols.get("feature") or df.columns[0]
            desc_col = cols.get("description") or cols.get("function_name") or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
            df[ko_col] = df[ko_col].astype(str).str.strip()
            df[desc_col] = df[desc_col].astype(str).str.strip()
            df = df[(df[ko_col] != "") & (df[desc_col] != "")]
            return dict(zip(df[ko_col], df[desc_col]))
    return {}


def load_kegg_matrix() -> pd.DataFrame:
    candidates = [
        MAASLIN_DIR / "KEGG_filtered_data_norm.tsv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, sep="\t", index_col=0)
            df.index = df.index.astype(str).str.strip()
            df.columns = df.columns.astype(str).str.strip()
            df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            return df.loc[~df.index.duplicated(), ~df.columns.duplicated()].copy()
    raise FileNotFoundError("KEGG matrix not found")


def load_genus_matrix() -> pd.DataFrame:
    df = pd.read_csv(RES / "Genus_count_matrix_12samples_PhuTho_SonLa.tsv", sep="\t", index_col=0)
    df.columns = df.columns.astype(str).str.strip()
    df.index = df.index.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rel = df.div(df.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    return rel.T


def summarize_feature_matrix(X: pd.DataFrame, y_host: pd.Series, y_status: pd.Series, feature_type: str) -> pd.DataFrame:
    keywords = {"sample", "host", "condition", "citrus", "coffee", "healthy", "disease", "diseased", "status"}
    sample_tokens = {s.lower() for s in X.index.astype(str)}
    leak_rows = []
    for col in X.columns.astype(str):
        col_low = col.lower()
        hit_keyword = any(tok in col_low for tok in keywords)
        hit_sample = col_low in sample_tokens
        exact_host = col_low in {v.lower() for v in y_host.astype(str).unique()}
        exact_status = col_low in {v.lower() for v in y_status.astype(str).unique()}
        if hit_keyword or hit_sample or exact_host or exact_status:
            leak_rows.append({"feature": col, "issue": "suspicious_name"})

    return pd.DataFrame({
        "feature_type": [feature_type],
        "n_samples": [int(X.shape[0])],
        "n_features": [int(X.shape[1])],
        "duplicate_columns": [int(X.columns.duplicated().sum())],
        "duplicate_rows": [int(X.index.duplicated().sum())],
        "non_numeric_cells": [int(pd.isna(X).sum().sum())],
        "suspicious_feature_names": [int(len(leak_rows))],
    }), pd.DataFrame(leak_rows)


def make_rf(seed: int = RANDOM_SEED, n_estimators: int = 700) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def loocv_auc(X: pd.DataFrame, y_raw: pd.Series, seed: int = RANDOM_SEED, n_estimators: int = 700):
    common = X.index.intersection(y_raw.index)
    X = X.loc[common].copy()
    y_raw = y_raw.loc[common].copy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    loo = LeaveOneOut()
    prob = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in loo.split(X):
        rf = make_rf(seed, n_estimators=n_estimators)
        rf.fit(X.iloc[train_idx], y[train_idx])
        prob[test_idx[0]] = rf.predict_proba(X.iloc[test_idx])[:, 1][0]
    auc_val = float(roc_auc_score(y, prob))
    fpr, tpr, _ = roc_curve(y, prob)
    rf_full = make_rf(seed, n_estimators=n_estimators)
    rf_full.fit(X, y)
    imp = pd.DataFrame({"feature": X.columns, "importance": rf_full.feature_importances_}).sort_values("importance", ascending=False)
    return {
        "classes": list(le.classes_),
        "auc": auc_val,
        "fpr": fpr,
        "tpr": tpr,
        "prob": pd.Series(prob, index=X.index, name="prob_positive"),
        "importance": imp,
        "y": pd.Series(y_raw, index=X.index, name="group"),
        "positive_label": le.classes_[1],
        "negative_label": le.classes_[0],
    }


def permutation_test_loocv(
    X: pd.DataFrame,
    y_raw: pd.Series,
    n_permutations: int = 200,
    seed: int = RANDOM_SEED,
    n_estimators_perm: int = 250,
):
    rng = np.random.default_rng(seed)
    observed = loocv_auc(X, y_raw, seed=seed)["auc"]
    perm_aucs = []
    y_values = y_raw.values.copy()
    for i in range(n_permutations):
        shuffled = rng.permutation(y_values)
        perm_res = loocv_auc(X, pd.Series(shuffled, index=y_raw.index), seed=seed + i + 1, n_estimators=n_estimators_perm)
        perm_aucs.append(perm_res["auc"])
    perm_aucs = np.array(perm_aucs, dtype=float)
    p_value = float((1 + np.sum(perm_aucs >= observed)) / (1 + len(perm_aucs)))
    return observed, perm_aucs, p_value


def nested_combined_loocv(
    Xkegg: pd.DataFrame,
    Xgenus: pd.DataFrame,
    y_status: pd.Series,
    top_k_kegg: int = TOP_K_KEGG,
    top_k_genus: int = TOP_K_GENUS,
    seed: int = RANDOM_SEED,
):
    common = Xkegg.index.intersection(Xgenus.index).intersection(y_status.index)
    Xkegg = Xkegg.loc[common].copy()
    Xgenus = Xgenus.loc[common].copy()
    y_raw = y_status.loc[common].copy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    loo = LeaveOneOut()
    prob = np.zeros(len(y), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(Xkegg), start=1):
        Xkegg_train = Xkegg.iloc[train_idx]
        Xgenus_train = Xgenus.iloc[train_idx]
        y_train_raw = y_raw.iloc[train_idx]
        y_train = y[train_idx]

        rf_kegg = make_rf(seed + fold_idx, n_estimators=MAIN_TREES)
        rf_kegg.fit(Xkegg_train, y_train)
        top_kegg = pd.Series(rf_kegg.feature_importances_, index=Xkegg_train.columns).sort_values(ascending=False).head(top_k_kegg).index.tolist()

        rf_genus = make_rf(seed + 1000 + fold_idx, n_estimators=MAIN_TREES)
        rf_genus.fit(Xgenus_train, y_train)
        top_genus = pd.Series(rf_genus.feature_importances_, index=Xgenus_train.columns).sort_values(ascending=False).head(top_k_genus).index.tolist()

        X_train_comb = pd.concat([
            Xkegg_train.loc[:, top_kegg].add_prefix("KEGG::"),
            Xgenus_train.loc[:, top_genus].add_prefix("Genus::"),
        ], axis=1)
        X_test_comb = pd.concat([
            Xkegg.iloc[test_idx].loc[:, top_kegg].add_prefix("KEGG::"),
            Xgenus.iloc[test_idx].loc[:, top_genus].add_prefix("Genus::"),
        ], axis=1)

        rf_comb = make_rf(seed + 2000 + fold_idx, n_estimators=MAIN_TREES)
        rf_comb.fit(X_train_comb, y_train)
        prob[test_idx[0]] = rf_comb.predict_proba(X_test_comb)[:, 1][0]

    auc_val = float(roc_auc_score(y, prob))
    fpr, tpr, _ = roc_curve(y, prob)
    return {
        "auc": auc_val,
        "fpr": fpr,
        "tpr": tpr,
        "prob": pd.Series(prob, index=common, name="prob_positive"),
        "y": pd.Series(y_raw, index=common, name="group"),
        "positive_label": le.classes_[1],
        "negative_label": le.classes_[0],
    }


def compute_direction_table(X: pd.DataFrame, y: pd.Series, importance: pd.DataFrame, positive_label: str, negative_label: str, label_map: dict[str, str] | None = None):
    pos_samples = y[y.astype(str) == positive_label].index
    neg_samples = y[y.astype(str) == negative_label].index
    mean_pos = X.loc[pos_samples].mean(axis=0)
    mean_neg = X.loc[neg_samples].mean(axis=0)
    out = importance.copy()
    out[f"mean_{positive_label.lower()}"] = out["feature"].map(mean_pos.to_dict())
    out[f"mean_{negative_label.lower()}"] = out["feature"].map(mean_neg.to_dict())
    out["favored_group"] = np.where(out[f"mean_{positive_label.lower()}"] >= out[f"mean_{negative_label.lower()}"] , positive_label, negative_label)
    out["signed_importance"] = np.where(out["favored_group"].eq(positive_label), out["importance"], -out["importance"])
    if label_map is not None:
        out["feature_label"] = out["feature"].map(lambda x: format_feature_label(x, label_map, width=72))
    return out


def format_feature_label(feature: str, ko_map: dict[str, str], width: int = 36) -> str:
    feature = str(feature)
    desc = ko_map.get(feature, "")
    label = f"{feature}: {desc}" if desc and desc != "-" else feature
    wrapped = textwrap.wrap(label, width=width)
    return "\n".join(wrapped[:2]) if wrapped else label


def plot_roc(ax, res: dict, title: str, note: str | None = None):
    ax.plot(res["fpr"], res["tpr"], color="#111111", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    text = f"LOOCV AUC={res['auc']:.3f}"
    if note:
        text = f"{text}\n{note}"
    ax.text(
        0.57,
        0.14,
        text,
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9),
    )


def plot_two_panel_roc(
    left_res: dict,
    right_res: dict,
    suptitle: str,
    left_title: str,
    right_title: str,
    out_paths: list[Path],
    left_note: str | None = None,
    right_note: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.98)
    plot_roc(axes[0], left_res, left_title, note=left_note)
    plot_roc(axes[1], right_res, right_title, note=right_note)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    for out_path in out_paths:
        fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_signed_importance(ax, table: pd.DataFrame, positive_label: str, negative_label: str, positive_color: str, negative_color: str, title: str, labels: list[str]):
    top = table.head(15).iloc[::-1].copy()
    colors = [positive_color if grp == positive_label else negative_color for grp in top["favored_group"]]
    ax.barh(range(len(top)), top["signed_importance"], color=colors, edgecolor="#2B2B2B", linewidth=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels[::-1], fontsize=8.5)
    ax.set_xlabel("Signed feature importance (Gini)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    lim = float(np.max(np.abs(top["signed_importance"]))) if len(top) else 1.0
    ax.set_xlim(-lim * 1.25, lim * 1.25)
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=positive_color, edgecolor="black", label=f"{positive_label}-enriched"),
            mpatches.Patch(facecolor=negative_color, edgecolor="black", label=f"{negative_label}-enriched"),
        ],
        loc="lower right",
        fontsize=8.5,
        framealpha=0.9,
    )


def plot_permutation_hist(observed: float, perm_aucs: np.ndarray, p_value: float, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(perm_aucs, bins=30, color="#C7D7E8", edgecolor="#385878")
    ax.axvline(observed, color="#D62728", lw=2.2, label=f"Observed AUC = {observed:.3f}")
    ax.set_xlabel("LOOCV AUC under label permutation", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.text(
        0.98,
        0.95,
        f"Permutation p = {p_value:.4f}\nN = {len(perm_aucs)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9),
    )
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_combined_status_model(Xkegg: pd.DataFrame, Xgenus: pd.DataFrame, y_status: pd.Series, ko_map: dict[str, str]):
    kegg_res = loocv_auc(Xkegg, y_status, seed=RANDOM_SEED, n_estimators=MAIN_TREES)
    genus_res = loocv_auc(Xgenus, y_status, seed=RANDOM_SEED, n_estimators=MAIN_TREES)

    top_kegg = kegg_res["importance"].head(TOP_K_KEGG)["feature"].tolist()
    top_genus = genus_res["importance"].head(TOP_K_GENUS)["feature"].tolist()

    common = Xkegg.index.intersection(Xgenus.index).intersection(y_status.index)
    X_comb = pd.concat([
        Xkegg.loc[common, top_kegg].add_prefix("KEGG::"),
        Xgenus.loc[common, top_genus].add_prefix("Genus::"),
    ], axis=1)
    y = y_status.loc[common].copy()
    comb_res = nested_combined_loocv(Xkegg.loc[common], Xgenus.loc[common], y, top_k_kegg=TOP_K_KEGG, top_k_genus=TOP_K_GENUS, seed=RANDOM_SEED)
    comb_fit_res = loocv_auc(X_comb, y, seed=RANDOM_SEED, n_estimators=MAIN_TREES)
    comb_dir = compute_direction_table(X_comb, comb_fit_res["y"], comb_fit_res["importance"], STATUS_POS, STATUS_NEG)
    comb_dir["feature_source"] = np.where(comb_dir["feature"].str.startswith("KEGG::"), "KEGG", "Genus")
    comb_dir["raw_feature"] = comb_dir["feature"].str.replace(r"^(KEGG|Genus)::", "", regex=True)
    comb_dir["feature_label"] = comb_dir.apply(
        lambda r: format_feature_label(r["raw_feature"], ko_map, width=56) if r["feature_source"] == "KEGG" else r["raw_feature"],
        axis=1,
    )
    return kegg_res, genus_res, X_comb, comb_res, comb_dir, top_kegg, top_genus


def plot_combined_status_figure(comb_res: dict, comb_dir: pd.DataFrame, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1.0, 1.35]})
    fig.suptitle("Combined Random Forest for Healthy vs Disease\n(Top KEGG + top Genus features, LOOCV)", fontsize=16, fontweight="bold", y=0.98)
    plot_roc(ax1, comb_res, "(A) ROC curve for combined model")
    top = comb_dir.head(15).copy()
    labels = [f"[{src}] {lab}" for src, lab in zip(top["feature_source"], top["feature_label"])]
    plot_signed_importance(ax2, comb_dir, STATUS_POS, STATUS_NEG, COL_STATUS[STATUS_POS], COL_STATUS[STATUS_NEG], "(B) Top 15 signed Gini features", labels)
    ax2.text(
        0.98,
        0.03,
        "Right/light green = Healthy-enriched\nLeft/orange = Disease-enriched",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print("Loading metadata and feature matrices...", flush=True)
    meta = load_metadata()
    ko_map = load_ko_map()
    Xkegg = load_kegg_matrix().loc[lambda d: d.index.intersection(meta.index)]
    Xgenus = load_genus_matrix().loc[lambda d: d.index.intersection(meta.index)]
    meta_kegg = meta.loc[Xkegg.index]
    meta_genus = meta.loc[Xgenus.index]

    summary_kegg, suspicious_kegg = summarize_feature_matrix(Xkegg, meta_kegg["host"], meta_kegg["condition"], "KEGG_17HQ")
    summary_genus, suspicious_genus = summarize_feature_matrix(Xgenus, meta_genus["host"], meta_genus["condition"], "Genus")
    summary_df = pd.concat([summary_kegg, summary_genus], ignore_index=True)
    summary_df.to_csv(OUT / "RF_input_matrix_summary.tsv", sep="\t", index=False)
    pd.concat([suspicious_kegg.assign(feature_type="KEGG_17HQ"), suspicious_genus.assign(feature_type="Genus")], ignore_index=True).to_csv(
        OUT / "RF_input_matrix_suspicious_features.tsv", sep="\t", index=False
    )

    print("Running Host LOOCV models...", flush=True)
    host_kegg_res = loocv_auc(Xkegg, meta_kegg["host"], seed=RANDOM_SEED, n_estimators=MAIN_TREES)
    host_genus_res = loocv_auc(Xgenus, meta_genus["host"], seed=RANDOM_SEED, n_estimators=MAIN_TREES)
    print(f"Running Host permutation tests ({N_PERMUTATIONS} permutations)...", flush=True)
    host_kegg_obs, host_kegg_perm, host_kegg_p = permutation_test_loocv(
        Xkegg,
        meta_kegg["host"],
        n_permutations=N_PERMUTATIONS,
        seed=RANDOM_SEED,
        n_estimators_perm=PERMUTATION_TREES,
    )
    host_genus_obs, host_genus_perm, host_genus_p = permutation_test_loocv(
        Xgenus,
        meta_genus["host"],
        n_permutations=N_PERMUTATIONS,
        seed=RANDOM_SEED + 1000,
        n_estimators_perm=PERMUTATION_TREES,
    )

    plot_permutation_hist(
        host_kegg_obs,
        host_kegg_perm,
        host_kegg_p,
        "Permutation test for Host RF (KEGG, LOOCV)",
        FIG / "RF_PermutationTest_Host_KEGG_LOOCV.png",
    )
    plot_permutation_hist(
        host_genus_obs,
        host_genus_perm,
        host_genus_p,
        "Permutation test for Host RF (Genus, LOOCV)",
        FIG / "RF_PermutationTest_Host_Genus_LOOCV.png",
    )

    host_validation = pd.DataFrame([
        {"feature_set": "KEGG_17HQ", "target": "Host_Citrus_vs_Coffee", "loocv_auc": round(host_kegg_obs, 4), "permutation_p": round(host_kegg_p, 6), "n_permutations": N_PERMUTATIONS},
        {"feature_set": "Genus", "target": "Host_Citrus_vs_Coffee", "loocv_auc": round(host_genus_obs, 4), "permutation_p": round(host_genus_p, 6), "n_permutations": N_PERMUTATIONS},
    ])
    host_validation.to_csv(OUT / "RF_Host_LOOCV_permutation_summary.tsv", sep="\t", index=False)

    print("Building combined Healthy vs Disease model...", flush=True)
    status_kegg_res, status_genus_res, X_comb, comb_res, comb_dir, top_kegg, top_genus = build_combined_status_model(Xkegg, Xgenus, meta.loc[Xkegg.index.intersection(Xgenus.index), "condition"], ko_map)
    comb_dir.to_csv(OUT / "RF_Combined_Status_Healthy_vs_Disease_importance_with_direction.tsv", sep="\t", index=False)
    pd.DataFrame({"feature_source": ["KEGG"] * len(top_kegg) + ["Genus"] * len(top_genus), "feature": top_kegg + top_genus}).to_csv(
        OUT / "RF_Combined_Status_selected_features.tsv", sep="\t", index=False
    )
    pd.DataFrame([
        {"model": "KEGG_status_LOOCV", "auc": round(status_kegg_res["auc"], 4), "n_features": int(Xkegg.shape[1])},
        {"model": "Genus_status_LOOCV", "auc": round(status_genus_res["auc"], 4), "n_features": int(Xgenus.shape[1])},
        {"model": "Combined_status_LOOCV", "auc": round(comb_res["auc"], 4), "n_features": int(X_comb.shape[1])},
    ]).to_csv(OUT / "RF_Combined_Status_LOOCV_summary.tsv", sep="\t", index=False)

    plot_two_panel_roc(
        host_genus_res,
        status_genus_res,
        "Supplementary Figure S1. Genus-level Random Forest ROC curves",
        "(A) Host classification from genus profiles",
        "(B) Health-status classification from genus profiles",
        [FIG / "RF_Genus_Host_Status_LOOCV.png", PUB / "Figure S1.png"],
        left_note=f"Permutation p={host_genus_p:.4f}",
    )
    plot_two_panel_roc(
        host_kegg_res,
        status_kegg_res,
        "Supplementary Figure S2. KEGG ortholog-level Random Forest ROC curves",
        "(A) Host classification from KEGG profiles",
        "(B) Health-status classification from KEGG profiles",
        [FIG / "RF_KEGG_Host_Status_LOOCV.png", PUB / "Figure S2.png"],
        left_note=f"Permutation p={host_kegg_p:.4f}",
    )

    plot_combined_status_figure(comb_res, comb_dir, FIG / "RF_Combined_Status_Healthy_vs_Disease_LOOCV.png")

    print(OUT / "RF_input_matrix_summary.tsv")
    print(OUT / "RF_Host_LOOCV_permutation_summary.tsv")
    print(OUT / "RF_Combined_Status_LOOCV_summary.tsv")
    print(FIG / "RF_Genus_Host_Status_LOOCV.png")
    print(FIG / "RF_KEGG_Host_Status_LOOCV.png")
    print(PUB / "Figure S1.png")
    print(PUB / "Figure S2.png")
    print(FIG / "RF_PermutationTest_Host_KEGG_LOOCV.png")
    print(FIG / "RF_PermutationTest_Host_Genus_LOOCV.png")
    print(FIG / "RF_Combined_Status_Healthy_vs_Disease_LOOCV.png")


if __name__ == "__main__":
    main()
