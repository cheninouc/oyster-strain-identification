# oyster-strain-identification

Analysis scripts for the machine-learning section of:

> Development of an optimized SNP panel and PCR-based INDEL markers for the identification of seven breeding strains of the Pacific oyster (*Crassostrea gigas*)

This repository contains the four-step pipeline that integrates population differentiation (FST) with PCA allele weights, performs stability selection with Elastic Net, optimizes the panel size, and validates the final panel in an independent cohort.

## Requirements

- PLINK 1.9 (`plink`) - VCF to BED conversion, LD pruning, `--recode A`
- PLINK 2.0 (`plink2`) - `--pca` with `allele-wts`
- VCFtools (`vcftools`) - per-strain Weir & Cockerham FST
- Python 3.9+ with `numpy`, `pandas`, `scikit-learn`, `joblib`

## Input files

All paths are configured in the `stepN_run.sh` scripts through the `WORKDIR` variable:

| File | Content |
| --- | --- |
| `vcf/all_snp.recode.vcf.gz` | Genome-wide SNP genotypes of the discovery set |
| `vcf/sample_pop` | Two columns: `IID<TAB>population` (discovery labels) |
| `step4/snp_subset.vcf` | Genotypes of the 125 independent validation individuals at the panel SNPs |
| `step4/news_sample_pop` | Two columns: `IID<TAB>population` (validation labels) |

Population codes used in the scripts are `CF`, `JF`, `KF`, `SB`, `SG`, `SO`, `SW` for the seven breeding strains (CS, JS, KS, BS, GS, OS, WS in the manuscript) and `Wild` for the wild population.

## Pipeline

### Step 1 - FST/PCA rank integration (`step1_run.sh`, `step1_rank_integration.py`)

1. Convert the VCF to BED and prune for LD (`--indep-pairwise 50 5 0.5`).
2. Run PLINK 2.0 PCA (`--pca 5 meanimpute allele-wts`) to obtain sample scores and per-SNP allele weights.
3. Compute target-vs-rest FST for each strain with VCFtools on the pruned SNP set.
4. Retain the FST top 1%, score each SNP by the weighted absolute PCA loading, rank FST and PCA scores, keep the PCA top 20%, and integrate the two normalized ranks by their geometric mean.
5. Output the top 2,000 SNPs per strain (`step1/out/<strain>_top2000.tsv`).

### Step 2 - One-vs-rest Elastic Net stability selection (`step2_run.sh`, `step2_elasticnet.py`)

For each strain, the 2,000 step-1 candidates are used as features; the strain is the positive class and the remaining six strains plus the wild population form the negative class. Hyperparameters (`C`, `l1_ratio`) are chosen by stratified 5-fold CV with the one-standard-error sparse-model rule, then fixed for 100 stratified subsampling runs (80% of individuals). SNPs are ranked by selection frequency and mean absolute coefficient, and the top 200 per strain are merged and deduplicated into the candidate union (`step2/out/candidates_union.tsv`, 1,400 candidates when no SNP is shared between strains).

### Step 3 - Multinomial stability ranking and panel-size optimization (`step3_run.sh`, `step3_elasticnet.py`)

A multinomial Elastic Net is fitted to the 1,400 candidates (seven breeding strains as classes; the wild population serves as background). After hyperparameter tuning and 100 subsampling runs, the candidates are ranked by selection frequency and coefficient magnitude. Cumulative top-N panels (N = 10-200 in steps of 10, refined at single-SNP resolution around the first passing size) are evaluated with repeated stratified 5-fold CV (10 repeats). The final panel is the smallest N for which macro-F1, balanced accuracy, MCC and minimum class recall all reach the target value (1.0). Outputs: `step3/out/ranking.tsv`, `panel_size_metrics.tsv`, `panel_class_recall.tsv`, `final_panel.tsv`.

### Step 4 - Independent validation (`step4_run.sh`, `step4_validate.py`)

The locked panel is used to train a multinomial logistic regression on the discovery individuals and to predict the 125 independent validation individuals. Alleles are aligned between the two PLINK runs before prediction: sites whose A1 allele differs are flipped (`2 - x`), and the flipped sites are recorded in `allele_alignment.tsv`. Outputs: `step4/out/predictions.tsv`, `confusion_matrix.tsv`, `confusion_rowpct.tsv`, `metrics.tsv`, `panel_dosages.tsv`.

## Running the pipeline

1. Edit the `WORKDIR` variable at the top of each `stepN_run.sh` script. The default value is the server directory used in the study and **must** be replaced with your own path.
2. Run the steps in order:

```bash
bash step1_run.sh
bash step2_run.sh
bash step3_run.sh
bash step4_run.sh
```

Each step reads the output of the previous one, so the order cannot be changed. Step 2 and step 3 are computationally intensive (100 subsampling fits per model); the scripts parallelize across strains, hyperparameter combinations and subsamples.

## Notes

- The random seed is fixed (`20261027`) so that the published run can be reproduced from the same input data.
- The panel size (85 SNPs) is data-dependent; other datasets may yield a different size.
- Log messages are printed in Chinese.

## Citation

If you use these scripts, please cite the accompanying manuscript:

Chen, C., Chen, D., Guo, C., Li, Q., Liu, S. Development of an optimized SNP panel and PCR-based INDEL markers for the identification of seven breeding strains of the Pacific oyster (*Crassostrea gigas*).
