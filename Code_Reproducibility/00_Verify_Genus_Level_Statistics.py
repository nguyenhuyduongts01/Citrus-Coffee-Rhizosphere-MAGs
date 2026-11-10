#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_Verify_Genus_Level_Statistics.py
===================================
Independent recomputation (no hard-coded values) of the genus-level statistics reported in Results 3.2 / Fig 2:
  * Bray-Curtis PCoA axis variance (PC1, PC2)
  * PERMANOVA (pseudo-F, R2, permutation p) for host/site and symptom status (log(1+x) genus counts, 12 samples)
  * Shannon entropy and observed genus richness, two-sided Mann-Whitney U tests

Input : Source_Data/results/adonis_counts.tsv (12 samples x 2,716 genera, Kraken2/Bracken counts)
Output: printed table + Source_Data/results/Genus_level_statistics_recomputed.tsv

Note: PERMANOVA F and R2 are deterministic and reproduce the manuscript values exactly. Permutation p-values depend on the
permutation scheme and random seed (manuscript: vegan::adonis2, 1,000 permutations; here 9,999 random permutations);
the host/site effect is significant (p < 0.01) and the symptom effect is not (p > 0.05) under both.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Source_Data" / "results" / "adonis_counts.tsv"
OUT = ROOT / "Source_Data" / "results"
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = [f"S{i}" for i in range(1, 13)]
HOST = np.array([0] * 6 + [1] * 6)  # S1-S6 citrus, S7-S12 coffee
SYMPTOM = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0])  # symptomatic: S1, S2, S9, S11

counts = pd.read_csv(SRC, sep="\t", index_col=0).loc[SAMPLES]
D = squareform(pdist(np.log1p(counts.values), "braycurtis"))


def permanova(D, g, n_perm=9999, seed=1):
    N = len(g)
    d2 = D ** 2
    sst = d2[np.triu_indices(N, 1)].sum() / N

    def stat(labels):
        ssw = 0.0
        for k in np.unique(labels):
            idx = np.where(labels == k)[0]
            ssw += d2[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].sum() / len(idx)
        a = len(np.unique(labels))
        return ((sst - ssw) / (a - 1)) / (ssw / (N - a)), (sst - ssw) / sst

    f0, r2 = stat(g)
    rng = np.random.default_rng(seed)
    hits = sum(stat(rng.permutation(g))[0] >= f0 for _ in range(n_perm))
    return f0, r2, (hits + 1) / (n_perm + 1)


rows = []
for name, g in [("host/site", HOST), ("symptom status", SYMPTOM)]:
    f, r2, p = permanova(D, g)
    rows.append({"analysis": f"PERMANOVA {name}", "statistic": f"F = {f:.3f}; R2 = {r2:.3f}; p = {p:.4f}"})

J = np.eye(12) - np.ones((12, 12)) / 12
ev = np.sort(np.linalg.eigvalsh(-0.5 * J @ (D ** 2) @ J))[::-1]
ev = ev[ev > 0]
rows.append({"analysis": "PCoA axes", "statistic": f"PC1 = {100*ev[0]/ev.sum():.1f}%; PC2 = {100*ev[1]/ev.sum():.1f}%"})

shannon = counts.apply(lambda r: -np.sum((r[r > 0] / r.sum()) * np.log(r[r > 0] / r.sum())), axis=1).values
richness = (counts > 0).sum(axis=1).values
for metric, v in [("Shannon", shannon), ("Observed genus richness", richness)]:
    p_host = mannwhitneyu(v[HOST == 0], v[HOST == 1]).pvalue
    p_sym = mannwhitneyu(v[SYMPTOM == 1], v[SYMPTOM == 0]).pvalue
    rows.append({"analysis": f"Mann-Whitney U, {metric}", "statistic": f"host/site p = {p_host:.3f}; symptom status p = {p_sym:.3f}"})

res = pd.DataFrame(rows)
res.to_csv(OUT / "Genus_level_statistics_recomputed.tsv", sep="\t", index=False)
print(res.to_string(index=False))
