#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00b_Verify_PERMANOVA_All_Layers.py
==================================
Recomputes, from the derived matrices in Source_Data/, all PERMANOVA statistics reported in the manuscript:

  Layer                         Input matrix (12 samples)                              Transform      Grouping source
  ----------------------------  -----------------------------------------------------  -------------  ----------------------------------
  Kraken2 genus (Fig 2C-D)      results/adonis_counts.tsv                              log(1+x)       Metadata of the 12-sample design
  KEGG / COG / CAZy (S4 Fig)    functional_matrices/*_combined_results.tsv             log(1+x)       results/Metadata_12samples_MaAsLin2.tsv
  62 MQ+HQ MAGs (sensitivity)   results/MAG_relative_abundance_89bac.tsv               none (rel.ab.) MAG abundance table (host / condition)

pseudo-F and R2 (= SS_between / SS_total) are deterministic and are compared with the reported values
(vegan::adonis2, 1,000 permutations, stored in results/PERMANOVA_adonis2_1000perm_results.tsv), alongside
the exact permutation p-value (full enumeration of all label assignments). The 62 MQ+HQ MAG sensitivity
analysis is evaluated with 9,999 Monte-Carlo permutations (seed 42).

Output: printed table + Source_Data/results/PERMANOVA_all_layers_summary.tsv
"""
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Source_Data"
RES = SRC / "results"
FUN = SRC / "functional_matrices"
OUT = SRC / "results"
OUT.mkdir(parents=True, exist_ok=True)


def pseudo_f_r2(D, g):
    g = np.asarray(g)
    n = len(g)
    d2 = D ** 2
    iu = np.triu_indices(n, 1)
    sst = d2[iu].sum() / n
    ssw = 0.0
    labs = np.unique(g)
    for k in labs:
        idx = np.where(g == k)[0]
        sub = d2[np.ix_(idx, idx)]
        ssw += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
    a = len(labs)
    return ((sst - ssw) / (a - 1)) / (ssw / (n - a)), (sst - ssw) / sst


def exact_p(D, g):
    """Exact permutation p-value (two groups) by enumerating every label assignment."""
    g = np.asarray(g)
    n = len(g)
    labs = np.unique(g)
    k = int((g == labs[0]).sum())
    f_obs = pseudo_f_r2(D, g)[0]
    hits = tot = 0
    for comb in itertools.combinations(range(n), k):
        gg = np.full(n, 1)
        gg[list(comb)] = 0
        tot += 1
        hits += pseudo_f_r2(D, gg)[0] >= f_obs - 1e-12
    return hits / tot


def mc_p(D, g, n_perm=9999, seed=42):
    """Monte-Carlo permutation p-value for the 62 MQ+HQ MAG sensitivity analysis."""
    g = np.asarray(g)
    rng = np.random.default_rng(seed)
    f_obs = pseudo_f_r2(D, g)[0]
    hits = sum(pseudo_f_r2(D, rng.permutation(g))[0] >= f_obs for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


reported = pd.read_csv(RES / "PERMANOVA_adonis2_1000perm_results.tsv", sep="\t").set_index(["layer", "factor"])
meta12 = pd.read_csv(RES / "Metadata_12samples_MaAsLin2.tsv", sep="\t").set_index("sample")
rows = []


def add(layer, factor, D, g, perm_p=None, note=""):
    f, r2 = pseudo_f_r2(D, g)
    row = {"layer": layer, "factor": factor, "pseudo_F_recomputed": round(f, 3), "R2_recomputed": round(r2, 3),
           "exact_permutation_p": round(exact_p(D, g), 4) if perm_p is None else None,
           "mc_p_9999_seed42": None if perm_p is None else round(perm_p, 4), "note": note}
    if (layer, factor) in reported.index:
        lk = reported.loc[(layer, factor)]
        row["pseudo_F_reported"] = lk["pseudo_F_reported"]
        row["p_reported"] = lk["p_reported"]
        row["F_matches_reported"] = abs(f - lk["pseudo_F_reported"]) < 0.0051
    rows.append(row)


# --- 1. Kraken2 genus level (12 samples) -------------------------------------------------------------------------
samples = [f"S{i}" for i in range(1, 13)]
counts = pd.read_csv(RES / "adonis_counts.tsv", sep="\t", index_col=0).loc[samples]
Dg = squareform(pdist(np.log1p(counts.values), "braycurtis"))
host = np.array(["Citrus"] * 6 + ["Coffee"] * 6)
symptom = np.array(["Symptomatic" if s in ("S1", "S2", "S9", "S11") else "Asymptomatic" for s in samples])
add("Genus", "host", Dg, host)
add("Genus", "condition", Dg, symptom, note="symptomatic = S1, S2, S9, S11 (Section 2.1)")

# --- 2. Functional profiles (KEGG / COG / CAZy), 12 samples -------------------------------------------------------
for layer in ("KEGG", "COG", "CAZy"):
    df = pd.read_csv(FUN / f"{layer}_combined_results.tsv", sep="\t", index_col=0)
    df.columns = [re.sub(r"_.*", "", c) for c in df.columns]
    cols = [c for c in samples if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).T
    D = squareform(pdist(np.log1p(X.values), "braycurtis"))
    m = meta12.loc[cols]
    add(layer, "host", D, m["host"].values)
    add(layer, "condition", D, m["condition"].values, note="grouping from Metadata_12samples_MaAsLin2.tsv")

# --- 3. 62 MQ+HQ MAG sensitivity analysis -------------------------------------------------------------------------
m62 = pd.read_csv(RES / "Metadata_62MQ_MAGs.tsv", sep="\t")
ab = pd.read_csv(RES / "MAG_relative_abundance_89bac.tsv", sep="\t")
ids = set(m62.loc[m62["pass_medium_quality"].astype(str).str.lower().eq("true"), "mag_clean"])
sub = ab[ab["mag_id"].isin(ids)]
wide = sub.pivot_table(index="sample", columns="mag_id", values="relative_abundance", fill_value=0)
mm = sub[["sample", "host", "condition"]].drop_duplicates().set_index("sample").loc[wide.index]
D62 = squareform(pdist(wide.values, "braycurtis"))
for factor in ("host", "condition"):
    g = mm[factor].values
    add("62MQ_MAGs", factor, D62, g, perm_p=mc_p(D62, g),
        note=f"{len(ids)} MQ+HQ MAGs ({wide.shape[1]} detected), Bray-Curtis on relative abundance")
stored = pd.read_csv(RES / "PERMANOVA_62MQ_MAGs_12samples.tsv", sep="\t").set_index("factor")
for r in rows:
    if r["layer"] == "62MQ_MAGs":
        r["pseudo_F_reported"] = round(stored.loc[r["factor"], "pseudo_F"], 3)
        r["p_reported"] = stored.loc[r["factor"], "p_value"]
        r["F_matches_reported"] = abs(r["pseudo_F_recomputed"] - r["pseudo_F_reported"]) < 0.0051

res = pd.DataFrame(rows)
res.to_csv(OUT / "PERMANOVA_all_layers_summary.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(res.drop(columns=["note"]).to_string(index=False))
assert res["F_matches_reported"].all(), "A recomputed pseudo-F differs from the reported value"
print("\nAll recomputed pseudo-F values match the reported values.")
