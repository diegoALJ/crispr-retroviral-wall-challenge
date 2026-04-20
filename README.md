# Retroviral Wall Challenge

This repository contains my solution and experimentation workflow for the **Retroviral Wall Challenge**, a Kaggle bioinformatics competition focused on predicting which **reverse transcriptases (RTs)** will be active for **prime editing** and ranking them by expected efficiency.

The central challenge is to build a model that **generalizes across evolutionary families**, rather than memorizing family identity. To address this, the competition uses **Leave-One-Family-Out (LOFO) cross-validation** and evaluates submissions with the **Cross-Lineage Score (CLS)** metric.

**Competition link:** (https://www.kaggle.com/competitions/retroviral-challenge-predict/overview) 


---

## Project Overview

Prime editing is one of the most precise genome editing technologies currently available. A reverse transcriptase (RT) fused to Cas9 nickase plays a key role in determining whether an edit succeeds, how efficiently it works, and in which cellular contexts it can be applied.

Most current prime editors rely on **MMLV-derived RTs**, which are relatively large and difficult to deliver therapeutically. However, nature contains thousands of alternative RTs across retroviruses, bacteria, retrotransposons, and mobile genetic elements. This competition explores whether computational models can identify promising RT candidates before costly wet-lab screening. 
The task is to predict, from an RT protein sequence and computed biophysical features, a **continuous score** indicating how likely the RT is to be active and how efficient it may be for prime editing. 

---


## Dataset

The dataset contains **57 experimentally tested reverse transcriptases** evaluated for prime editing activity. Each RT includes sequence information, family metadata, handcrafted biophysical features, embeddings, and predicted structures. 

### Main files

- **`train.csv`**  
  Full training data with:
  - `rt_name`
  - `sequence`
  - `active`
  - `pe_efficiency_pct`
  - `rt_family`
  - `protein_length_aa`
  - 66 handcrafted features

- **`test.csv`**  
  Same RT entries without `active` and `pe_efficiency_pct`

- **`sample_submission.csv`**  
  Example submission format

- **`esm2_embeddings.npz`**  
  Mean-pooled ESM-2 embeddings (1280 dimensions)

- **`feature_dictionary.csv`**  
  Description of handcrafted features

- **`family_splits.csv`**  
  Family membership breakdown

- **`structures.zip`**  
  Predicted 3D structures in PDB format for all RTs 
---

## Modeling Goal

The objective is to produce a **continuous `predicted_score`** for each RT:

- higher score = more likely to be active
- higher score = more likely to be efficient for prime editing

This score is used both for:

- **classification**, through **PR-AUC**
- **ranking**, through **Weighted Spearman** 
---

## Evaluation Metric

The competition uses **CLS (Cross-Lineage Score)**, defined as the harmonic mean of:

- **PR-AUC**
- **Weighted Spearman correlation** 

This means a good solution must do both:
1. separate active from inactive RTs
2. rank the strongest RTs near the top

A model that performs well on only one of these components will still obtain a weak CLS score. 

---

## Validation Strategy

This project follows the competition’s required **LOFO (Leave-One-Family-Out)** evaluation protocol:

- hold out one evolutionary family at a time
- train on the remaining families
- predict on the held-out family
- pool all out-of-fold predictions
- compute CLS on the complete pooled prediction vector 

This is critical because standard random folds would overestimate performance by allowing family-specific patterns to leak across train and validation sets.

---

## Repository Structure

```
retroviral-wall-challenge/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml
└── notebooks/
    ├── 01_eda.ipynb
    └── 02_modeling.ipynb

```

---

## Notebook roles
01_eda.ipynb
Exploratory data analysis, dataset inspection, family distributions, feature behavior, missing values, and competition-specific risks such as family memorization.
02_modeling.ipynb
Feature preparation, LOFO validation, model training, evaluation, and generation of submission.csv.

---

## Project Focus

This repository is intentionally notebook-centered because the competition explicitly requires shareable notebook-based code for review and Phase 2 consideration. The goal is to keep the workflow transparent, reproducible, and easy to inspect.

A more modular refactor into src/ may be added later, but for the competition submission stage the notebook-first format is the most practical and aligned with the challenge requirements.

---

## Reproducibility

To reproduce the experiments:

Join the Kaggle competition and accept the rules.
Download the provided files.
Place the data in the expected local or Kaggle directory structure.
Run the EDA notebook.
Run the modeling notebook to generate out-of-fold predictions and submission.csv.

---

## Important Modeling Notes

The competition explicitly warns that some feature groups can make models overfit evolutionary lineage rather than biological function:

FoldSeek similarity features may correlate strongly with family identity
ESM-2 embeddings can encode family membership very well
missing structural values should not automatically be interpreted as zeros

For that reason, careful LOFO validation is essential.

---

## Disclaimer

This repository is an independent portfolio and competition project based on a public Kaggle challenge and external wet-lab validation framework. All competition rules, data rights, and prize decisions belong to the official organizers and sponsor.

---

## References

Competition and dataset information are based on the official Kaggle challenge description and dataset documentation provided by the organizers.

Mandrake Bio. Retroviral Wall Challenge. https://kaggle.com/competitions/retroviral-challenge-predict, 2026. Kaggle.
