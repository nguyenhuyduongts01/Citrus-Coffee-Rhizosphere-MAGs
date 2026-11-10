# Reproducibility Code Package: Vietnamese Citrus and Coffee Rhizosphere Metagenomics

**Manuscript Title:** *Genome-resolved metagenomics characterizes novel bacterial diversity, metabolic potential, and host/site- and symptom-associated candidate biomarkers in citrus and coffee rhizospheres in Vietnam*  
**Journal:** *PLOS ONE*

---

## Overview

This repository provides all custom Python and R scripts (`Code_Reproducibility/`) and derived input data matrices (`Source_Data/`) required to reproduce the statistical analyses, main figures (`Fig1`–`Fig8`), and supplementary figures (`S1_Fig`–`S7_Fig`) reported in the manuscript.

All scripts resolve their input files directly from `Source_Data/` via relative paths and export publication-grade 600 DPI figures (`.png` and LZW-compressed RGB `.tif`) into `Figures/` (and `Supplementary_Figures/`).

---

## Script Inventory (`Code_Reproducibility/`)

| Script | Language | Description & Outputs |
|---|---|---|
| `00_Verify_Genus_Level_Statistics.py` | Python | Recomputes Kraken2/Bracken genus-level Bray–Curtis PCoA axis variances (`PC1 = 33.5%`, `PC2 = 16.2%`), PERMANOVA (`pseudo-F`, `R²`, permutation `p`), and two-sided Mann–Whitney *U* tests for Shannon diversity and observed genus richness (Results 3.2; `Fig 2`). |
| `00b_Verify_PERMANOVA_All_Layers.py` | Python | Recomputes Bray–Curtis PERMANOVA (`pseudo-F`, `R²`, and exact/Monte-Carlo permutation `p`-values) across all five analytical layers: Kraken2 genus (`Fig 2C–D`), KEGG orthologs, COG categories, CAZyme families (`S4 Fig`), and the 62 MQ+HQ MAG sensitivity analysis (Results 3.2). |
| `01_Figure1_panels_and_composite.py` | Python | Generates `Fig 1A` (phylum-level relative abundance of 89 bacterial MAGs), `Fig 1C` (CheckM2 completeness vs. contamination of 17 HQ MAGs), `Fig 1D` (COG category distribution), and assembles the 4-panel composite `Fig1.tif` / `Fig1.png`. |
| `01b_Figure1B_Phylogeny_89bac.R` | R (`ape`) | Renders the circular maximum-likelihood phylogenomic cladogram of 89 bacterial MAGs (`bac120` marker alignment, IQ-TREE 2 `LG+F+G4`) with four concentric metadata rings (`Fig 1B`). |
| `01b_Figure1B_Phylogeny_89bac.py` | Python | Python (`Biopython`/`matplotlib`) implementation of the circular phylogenomic cladogram (`Fig 1B`). |
| `02_Figure2_Alpha_Beta_Diversity.py` | Python | Generates `Fig2.tif` / `Fig2.png` (Observed genus richness, Shannon index, and Bray–Curtis PCoA with PERMANOVA `F`, `R²`, and `p`-values across host/site and symptom status). |
| `03_Figure3_MaAsLin2_KEGG_KO.py` | Python | Generates `Fig3.tif` / `Fig3.png` (Top 25 differentially abundant KEGG Orthologs identified by MaAsLin2 across host/site and symptom status). |
| `04_Figure4_RF_KEGG_KO.py` | Python | Generates `Fig4.tif` / `Fig4.png` (700-tree Leave-One-Out Cross-Validation Random Forest ROC curves and top 25 discriminating KEGG Orthologs by Mean Decrease Gini). |
| `05_06_Figure5_Figure6_Species_Sankey_Bubble.py` | Python | Generates `Fig5.tif` (host/site-associated species) and `Fig6.tif` (symptom-status-associated species), combining two-column Sankey diagrams (with unified phylum color strips) and MaAsLin2 bubble plots. |
| `07_Figure7_RF_Species.py` | Python | Generates `Fig7.tif` / `Fig7.png` (700-tree LOOCV Random Forest species-level ROC curves and top 25 discriminating species by Mean Decrease Gini). |
| `08_Figure8_HQ17_Dotplot_and_Chord.py` | Python | Generates `Fig8.tif` / `Fig8.png` (`Fig 8A`: functional gene dot-plot across 17 HQ MAGs; `Fig 8B`: 3-column bipartite chord diagram of top 30 marker genes with outgoing link counts `(n)`). |
| `09_Supplementary_Figures_S1_to_S7.py` | Python | Generates all seven supplementary figures (`S1_Fig.tif` through `S7_Fig.tif`). |
| `rf_validation_and_combined_loocv.py` | Python | Core Random Forest 700-tree LOOCV classification and permutation importance module imported by `04_Figure4_RF_KEGG_KO.py`. |
| `Phylum_Color_Palette.tsv` | TSV | Unified hex color palette for bacterial and archaeal phyla shared across `Fig 1`, `Fig 5`, `Fig 6`, `Fig 8`, `S5_Fig`, and `S6_Fig`. |

---

## Input Data Structure (`Source_Data/`)

- `Source_Data/results/`: Taxonomic count matrices (`adonis_counts.tsv`, `Genus_count_matrix_12samples_PhuTho_SonLa.tsv`), alpha/beta diversity tables (`Alpha_diversity_12samples_PhuTho_SonLa.tsv`, `PCoA_genus_BrayCurtis_12samples_coordinates.tsv`), MAG metadata and abundance matrices (`Metadata_89bac_MAGs.tsv`, `Metadata_62MQ_MAGs.tsv`, `Metadata_17HQ_MAGs.tsv`, `MAG_relative_abundance_89bac.tsv`, `MAG_Phylum_relative_abundance_89bac_12samples.tsv`), phylogenomic tree (`Phylogenetic_Tree_89bac_MAGs.nwk`), and functional annotation summaries (`COG_analysis_22_MAGs.tsv`, `Table_17MAGs_Marker_Genes_Summary.tsv`, `Chord_top10_MAG_specific_gene_links_HQ17_corrected.tsv`, `KEGG_KO_mapping.tsv`, `ko_description_map.tsv`).
- `Source_Data/functional_matrices/`: Sample-by-feature functional abundance matrices (`KEGG_combined_results.tsv`, `COG_combined_results.tsv`, `CAZy_combined_results.tsv`) used for `S4_Fig` and PERMANOVA.
- `Source_Data/P1_inputs/`: Kraken2/Bracken species long table (`Taxonomy_long_with_species_12samples.tsv`) and MaAsLin2 differential abundance outputs (`MaAsLin2_KEGG_pval005_annotated.tsv`, `species_lm_host.tsv`, `species_lm_cond.tsv`).
- `Source_Data/MaAsLin2/`: Normalized KEGG KO matrix (`KEGG_filtered_data_norm.tsv`), CAZyme MaAsLin2 results (`CAZyme_all_results.tsv`), and 700-tree LOOCV Random Forest signed importance tables (`RF_Host_KEGG_signed_importance.tsv`, `RF_Status_KEGG_signed_importance.tsv`).

---

## Software Requirements & Usage

- **Python** >= 3.10 (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `biopython`, `Pillow`)
- **R** >= 4.2 (`ape`, `dplyr`, `tidyr`, `RColorBrewer`)

To reproduce all statistical tables and figures from the repository root:

```bash
pip install -r requirements.txt
python3 Code_Reproducibility/00_Verify_Genus_Level_Statistics.py
python3 Code_Reproducibility/00b_Verify_PERMANOVA_All_Layers.py
python3 Code_Reproducibility/01b_Figure1B_Phylogeny_89bac.py
python3 Code_Reproducibility/01_Figure1_panels_and_composite.py
python3 Code_Reproducibility/02_Figure2_Alpha_Beta_Diversity.py
python3 Code_Reproducibility/03_Figure3_MaAsLin2_KEGG_KO.py
python3 Code_Reproducibility/04_Figure4_RF_KEGG_KO.py
python3 Code_Reproducibility/05_06_Figure5_Figure6_Species_Sankey_Bubble.py
python3 Code_Reproducibility/07_Figure7_RF_Species.py
python3 Code_Reproducibility/08_Figure8_HQ17_Dotplot_and_Chord.py
python3 Code_Reproducibility/09_Supplementary_Figures_S1_to_S7.py
```

---

## License

Code in this repository is released under the **MIT License** (`LICENSE`). Derived data tables in `Source_Data/` are shared under **CC-BY 4.0**.
