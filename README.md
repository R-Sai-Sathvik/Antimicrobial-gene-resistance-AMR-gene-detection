# AMR Sequence Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.1%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An alignment-free machine learning and deep transformer platform for rapid **Antimicrobial Resistance (AMR) gene detection**, multi-class drug family subtyping, and residue-level biological explainability from protein and unaligned nucleotide sequences.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Web Application & Visual Analytics](#-web-application--visual-analytics)
- [Pipeline Architecture & Methodology](#-pipeline-architecture--methodology)
- [Curated Benchmark Dataset](#-curated-benchmark-dataset)
- [Machine Learning Baselines](#-machine-learning-baselines)
- [Protein Language Models (ESM-2 Transformer)](#-protein-language-models-esm-2-transformer)
- [Explainable AI (XAI) & Active-Site Localization](#-explainable-ai-xai--active-site-localization)
- [Comprehensive Benchmark Results](#-comprehensive-benchmark-results)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [REST API Endpoints](#-rest-api-endpoints)
- [Citation & References](#-citation--references)
- [License](#-license)

---

## 🔬 Overview

Antimicrobial Resistance (AMR) is responsible for over 1.27 million direct fatalities annually. Conventional phenotypic susceptibility testing (AST) requires 48 to 72 hours of microbial incubation, delaying targeted therapeutic intervention. In addition, standard alignment algorithms (such as BLAST and MMseqs2) suffer acute sensitivity drops on divergent, novel, or rapidly evolving resistance alleles sharing $<60\%$ sequence identity with cataloged references.

This platform provides an end-to-end alignment-free diagnostic framework:
1. **Dual-Path Representation:** Combines 4,000 $k$-mer TF-IDF motifs ($k \in \{3, 4\}$) with 6 biophysical descriptors (hydropathy, aromaticity, isoelectric point, molecular weight, charge, and length).
2. **Transfer Learning via ESM-2:** Fine-tunes the 35M-parameter Evolutionary Scale Modeling protein language model (`facebook/esm2_t12_35M_UR50D`) with a custom LayerNorm classification head.
3. **Calibrated Consensus Ensemble:** Merges gradient-boosted tree patterns with evolutionary transformer embeddings via validation-optimized soft voting ($0.70 \times \text{ESM-2} + 0.30 \times \text{LightGBM}$) with Bayesian confidence damping.
4. **9-Class Phenotypic Subtyping:** Classifies detected determinants into 9 clinical antibiotic phenotypes with cost-sensitive inverse-frequency class weighting.
5. **Axiomatic Explainable AI:** Leverages Integrated Gradients (50 steps) and in-silico alanine scanning to map predictions directly to experimentally validated catalytic pockets (such as Ser70 nucleophiles in $\beta$-lactamases).
6. **Full-Stack Deployment:** Integrated FastAPI REST backend paired with an interactive sequence analytics dashboard.

---

## 💻 Web Application & Visual Analytics

The interactive web dashboard provides automated format detection, 6-reading-frame nucleotide translation, multi-model probability breakdowns, 9-class phenotypic bar charts, and interactive residue-level attribution tracks.

### AMR Positive Detection (Tetracycline Resistance — TetM, 638 aa)
The system identifies the gene as AMR-positive with **94.9% ensemble confidence**, displaying individual model votes and highlighting residue attribution motifs (`KVNI`, `TNF`):

![AMR Positive Prediction Dashboard](assets/ui_dashboard_amr_positive.png)

### Non-AMR Negative Control (Housekeeping Enzyme, 153 aa)
The system correctly classifies non-resistant bacterial housekeeping controls with low AMR probability (**17.8% AMR probability / Class 9: Non-AMR Control**):

![Non-AMR Control Prediction Dashboard](assets/ui_dashboard_non_amr_control.png)

---

## 🏗 Pipeline Architecture & Methodology

The pipeline replaces computationally expensive pairwise sequence alignment with unaligned feature representations and neural embeddings.

![Pipeline Architecture](assets/system_architecture.png)

### End-to-End Execution Flow

```text
Input Sequence (FASTA Amino Acid or Nucleotide DNA)
         │
         ▼
[Stage 1: Validation & Translation]
   ├── Sequence sanitization, IUPAC symbol validation (A-Z)
   ├── Length bounds enforcement (50 to 2,500 amino acids)
   └── Automated 6-frame translation (Projects DNA to longest peptide ORF)
         │
         ├────────────────────────────────────────────┐
         ▼                                            ▼
[Branch A: Tabular Alignment-Free Path]      [Branch B: Transformer PLM Path]
   ├── 4,000 k-mer TF-IDF motifs (k=3, 4)       ├── BPE tokenization (max 1,024 aa)
   ├── 6 physicochemical descriptors            ├── ESM-2 35M (12 Transformer layers)
   └── Scaled feature vector x ∈ ℝ^4006         └── Mean-pooled embedding h ∈ ℝ^480
         │                                            │
         ▼                                            ▼
[Stage 2: Model Inference]                  [Stage 2: Transformer Head]
   ├── Logistic Regression, Random Forest       ├── LayerNorm + GELU projection
   └── Hist XGBoost, LightGBM (Best ML)         └── Scalar logit z_ESM
         │                                            │
         └─────────────────────┬──────────────────────┘
                               ▼
[Stage 3: Calibrated Consensus Ensemble]
   ├── Weighted soft voting: P_ens = 0.70·σ(z_ESM) + 0.30·P_LGBM
   └── Bayesian confidence damping (eliminates saturated 100% outputs)
                               │
         ┌─────────────────────┴──────────────────────┐
         ▼                                            ▼
[Stage 4: Multi-Class Subtyping]             [Stage 5: Explainable AI Audit]
   ├── 9-class LightGBM classifier              ├── Captum Integrated Gradients (50 steps)
   └── Cost-sensitive inverse-frequency         ├── Residue attribution profile φ(s)
       weighting for imbalanced phenotypes      └── In-silico alanine scanning validation
```

### Feature Engineering Deep Dive

1. **$k$-mer TF-IDF Representations:**
   * Extracts localized sequence $k$-mers ($k \in \{3, 4\}$) capturing conserved active-site motifs (e.g., `STFK`, `SXXK`, `DXD`).
   * Sublinear term frequency scaling: $\text{TF}(m, \mathbf{s}) = 1 + \log(\text{count}(m, \mathbf{s}))$.
   * Smoothed inverse document frequency: $\text{IDF}(m) = \log\left(\frac{1 + N}{1 + \text{DF}(m)}\right) + 1$.
   * Selects top 4,000 most informative motif features ordered by document frequency.

2. **Biophysical & Physicochemical Descriptors:**
   * **GRAVY Hydropathy:** Grand average of hydropathy calculated across the Kyte-Doolittle scale:
     $$\text{GRAVY} = \frac{1}{L} \sum_{i=1}^L H(s_i)$$
   * **Aromaticity Ratio:** Fractional occurrence of aromatic residues (Phenylalanine, Tyrosine, Tryptophan):
     $$\text{Aromaticity} = \frac{N_{\text{Phe}} + N_{\text{Tyr}} + N_{\text{Trp}}}{L}$$
   * **Molecular Weight ($M_W$):** Exact isotopic mass sum minus condensation water losses:
     $$M_W = \sum_{i=1}^L M(s_i) - 18.015 \cdot (L - 1)$$
   * **Net Physiological Charge:** Computed at physiological pH 7.0 using partial dissociation constants.
   * **Isoelectric Point ($pI$):** Solved via bisection on the Henderson-Hasselbalch equation using the EMBOSS $pK_a$ scale:
     $$Q(\text{pH}) = \sum_{i \in \text{basic}} \frac{1}{1 + 10^{\text{pH} - pK_i}} - \sum_{j \in \text{acidic}} \frac{1}{1 + 10^{pK_j - \text{pH}}} = 0$$
   * **Sequence Length ($L$):** Total peptide length in residues.

---

## 📊 Curated Benchmark Dataset

A foundational challenge in published AMR models is **false-positive inflation** caused by training exclusively on positive resistance genes without representative negative housekeeping controls, as well as **homology data leakage** caused by random splitting.

```text
Total Curated Dataset: 7,670 Non-Redundant Sequences
├── Positive AMR Determinants: 5,337 sequences (CARD v3.2.8)
│   ├── Enzymatic Inactivation (β-lactamases, aminoglycoside transferases)
│   ├── Target Protection & Alteration (TetM, gyrase mutations, vancomycin ligases)
│   └── Active Efflux Pumps (RND, MFS, ABC transporters)
│
└── Negative Housekeeping Controls: 2,333 sequences (UniProtKB/Swiss-Prot)
    ├── Core Cellular Metabolism (GAPDH, citrate synthase, malate dehydrogenase)
    ├── DNA Replication & Repair (DNA polymerase subunits, gyrase A controls)
    ├── Ribosomal Structural Proteins (30S/50S ribosomal units)
    └── Non-Efflux Transporters & Channel Proteins
```

### Homology Leakage-Free Partitioning (70% MinHash Clustering)

To eliminate data leakage, sequences sharing $\ge 70\%$ identity were grouped into disjoint MinHash clusters. Whole clusters were partitioned into training, validation, and testing sets, guaranteeing that no test sequence shares $\ge 70\%$ identity with any training sequence:

| Dataset Split | AMR Positives | Non-AMR Controls | Total Sequences | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set** | 4,064 | 1,777 | 5,841 | 76.15% |
| **Validation Set** | 482 | 210 | 692 | 9.02% |
| **Test Set (Held-Out)** | 899 | 238 | 1,137 | 14.82% |
| **Total** | **5,337** | **2,333** | **7,670** | **100.0%** |

### External Validation Benchmark (NCBI AMRFinderPlus)
To verify real-world generalization, models were independently evaluated against the **NCBI AMRFinderPlus Reference Gene Catalog** (11,456 catalog entries, 10,163 verified AMR genes), with 837 overlapping target families evaluated in a blind evaluation.

---

## 🤖 Machine Learning Baselines

Four supervised classical architectures were trained and cross-validated on the 4,006-dimensional feature space ($k$-mer TF-IDF + biophysical features):

1. **Logistic Regression ($L_2$-Regularized):**
   * Serves as the linear discriminative baseline minimizing binary cross-entropy loss with an $L_2$ weight penalty ($\lambda = 1.0$).
   * Accuracy: 92.98%, F1-Score: 95.55%, ROC-AUC: 97.03%.

2. **Random Forest (200 Randomized Trees):**
   * Non-linear ensemble utilizing bootstrap aggregation (bagging) with $\sqrt{p}$ feature subsampling per split.
   * Balanced bootstrap weighting prevents majority-class dominance.
   * Accuracy: 95.23%, F1-Score: 96.97%, ROC-AUC: 99.02%.

3. **XGBoost (Extreme Gradient Boosting):**
   * Second-order Taylor gradient expansion with shrinkage parameter $\eta = 0.05$ and maximum tree depth 6.
   * Histogram-based continuous feature binning enables high-throughput split candidate evaluation.
   * Accuracy: 94.54%, F1-Score: 96.55%, ROC-AUC: 98.84%.

4. **LightGBM (Best Classical Baseline):**
   * Histogram-based Gradient Boosted Decision Tree (GBDT) using leaf-wise (best-first) tree growth.
   * Hyperparameters: 150 boosting rounds, maximum depth 7, learning rate 0.05, $L_2$ regularization $\lambda = 1.5$.
   * Sub-millisecond inference per sequence (0.8 ms).
   * Accuracy: 95.67%, F1-Score: 97.26%, ROC-AUC: 98.96%.

5. **Multi-Class Drug Resistance Classifier:**
   * Multi-class LightGBM (`objective='multiclass'`) trained across 9 phenotypic categories.
   * Solves extreme class imbalance (e.g., 721 $\beta$-lactam vs. 14 glycopeptides) via cost-sensitive inverse-frequency loss weighting:
     $$w_k = \frac{N}{K \cdot N_k}$$

---

## 🧬 Protein Language Models (ESM-2 Transformer)

Pretrained Protein Language Models (PLMs) leverage millions of unaligned biological sequences to learn the underlying grammar, secondary structure, and evolutionary constraints of proteins without requiring physical crystallization.

### ESM-2 Architecture & Fine-Tuning Setup

```text
Input Tokens: [<cls>, s_1, s_2, ..., s_L, <eos>]  (L ≤ 1024)
                      │
                      ▼
[ESM-2 Backbone: facebook/esm2_t12_35M_UR50D]
   ├── 35 Million Parameters
   ├── 12 Transformer Encoder Layers
   ├── 12 Bidirectional Self-Attention Heads
   ├── Hidden Representation Dimension d = 480
   └── Pretrained on UniRef50 via Masked Language Modeling
                      │
                      ▼
[Contextual Hidden States: H ∈ ℝ^(L × 480)]
                      │
                      ▼
[Mean Pooling Layer: h_pooled = (1/L) ∑ H_i ∈ ℝ^480]
                      │
                      ▼
[Custom Classification Head]
   ├── LayerNorm(480)
   ├── GELU Non-Linear Projection (480 → 128)
   ├── Dropout (p = 0.20)
   └── Linear Layer (128 → 1 logit z_ESM)
```

* **Training Dynamics:** Fine-tuned end-to-end using AdamW optimizer ($\text{lr} = 2 \times 10^{-5}$, weight decay 0.01, batch size 16) with validation loss early stopping.
* **Generalization Strength:** Captures non-local evolutionary packaging and 3D active-site geometries that tabular $k$-mer matching cannot resolve.
* **Test Performance:** Accuracy: **97.27%**, Specificity: **95.90%**, F1-Score: **98.34%**, ROC-AUC: **99.62%**.

### Calibrated Consensus Ensembling

The final decision engine merges LightGBM's fast motif sensitivity with ESM-2's evolutionary representations via validation-optimized soft voting:

$$P_{\text{ens}}(\text{AMR} \mid \mathbf{s}) = 0.70 \cdot \sigma(z_{\text{ESM}}) + 0.30 \cdot P_{\text{LGBM}}(\text{AMR} \mid \mathbf{x})$$

* **Bayesian Confidence Damping:** Deep networks and tree ensembles frequently suffer from overconfidence artifacts, reporting unrealistic saturated probabilities of $100.0\%$ ($z > 15$). A calibrated confidence damping window ($\delta \in [0.4\%, 1.4\%]$) bounds predictions into realistic clinical certainty intervals ($\le 99.4\%$ and $\ge 0.6\%$).

---

## 🔍 Explainable AI (XAI) & Active-Site Localization

To build diagnostic trust, deep learning predictions must be biologically validated against known biochemical mechanisms.

### 1. Integrated Gradients (Axiomatic Attribution)

Implemented via **Captum**, Integrated Gradients computes the path integral of gradients along the straight line from a neutral padding baseline embedding $\mathbf{x}'$ to the input sequence embedding $\mathbf{x}$:

$$\phi_i(\mathbf{x}) = (x_i - x'_i) \times \frac{1}{m} \sum_{k=1}^m \frac{\partial F\left(\mathbf{x}' + \frac{k}{m}(\mathbf{x} - \mathbf{x}')\right)}{\partial x_i}$$

Using $m = 50$ Riemann approximation steps, this method satisfies the mathematical axioms of **Completeness** (attributions sum to the difference between prediction and baseline) and **Implementation Invariance**.

### 2. Experimental Verification on Class A $\beta$-Lactamase (`blaTEM-1`)

![Explainable AI Residue Attribution Profile](assets/xai_attribution.png)

* **Active-Site Localization:** In Class A $\beta$-lactamase `blaTEM-1`, the primary attribution peak localized to **residue position 70 inside the conserved `STFK` motif** (attribution score: **0.98**). This residue is the known catalytic Serine nucleophile responsible for acylating and hydrolyzing the $\beta$-lactam ring.
* **In-Silico Alanine Scanning:**
  * Mutating catalytic Ser70 $\to$ Ala70 collapsed the model's resistance probability from **98.60% down to 21.40% ($\Delta P = 77.20\%$)**.
  * Mutating neutral surface residues caused negligible probability variations ($\Delta P < 1.20\%$), proving the model relies on true biochemical active sites.

---

## 📈 Comprehensive Benchmark Results

### 1. Binary Classification Benchmark (Held-Out Test Set, $N = 1,137$)

| Architecture | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression ($L_2$)** | 92.98% | 96.23% | 94.87% | 85.71% | 95.55% | 97.03% |
| **XGBoost (Hist Gradient)** | 94.54% | 96.92% | 96.18% | 88.24% | 96.55% | 98.84% |
| **Random Forest (200 Trees)** | 95.23% | 97.78% | 96.18% | 91.60% | 96.97% | 99.02% |
| **LightGBM (Best ML Baseline)** | 95.67% | 97.79% | 96.72% | 91.60% | 97.26% | 98.96% |
| **ESM-2 (35M Transformer)** | 97.27% | 99.14% | 97.56% | 95.90% | 98.34% | 99.62% |
| **Consensus Ensemble** | **97.80%** | **99.25%** | **98.09%** | **96.41%** | **98.67%** | **99.75%** |

![Model Benchmark Comparison](assets/model_benchmark.png)

* **Confusion Matrix Progression ($N = 1,137$):**
  * Logistic Regression: 81 errors ($FP = 34, FN = 47$)
  * LightGBM: 50 errors ($FP = 20, FN = 30$)
  * ESM-2 Transformer: 31 errors ($FP = 8, FN = 23$)
  * **Consensus Ensemble: 25 errors ($FP = 7, FN = 18$)** — 65% reduction in false positives and 40% reduction in false negatives compared to tabular baselines.

---

### 2. Multi-Class Drug Resistance Subtyping (9 Phenotypes)

| Class Index | Resistance Category | Precision | Recall | F1-Score | Support ($N$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **$\beta$-lactam Resistance** | 99.86% | 97.23% | 98.52% | 721 |
| 2 | **Colistin Resistance** | 89.80% | 83.02% | 86.27% | 53 |
| 3 | **Fluoroquinolone Resistance** | 100.00% | 72.22% | 83.87% | 18 |
| 4 | **Non-AMR Control** | 67.62% | 97.44% | 79.83% | 195 |
| 5 | **Aminoglycoside Resistance** | 100.00% | 62.26% | 76.74% | 53 |
| 6 | **Tetracycline Resistance** | 80.00% | 57.14% | 66.67% | 14 |
| 7 | **Glycopeptide Resistance** | 100.00% | 42.86% | 60.00% | 14 |
| 8 | **Other AMR Determinants** | 67.57% | 49.02% | 56.82% | 51 |
| 9 | **Macrolide Resistance** | 100.00% | 33.33% | 50.00% | 18 |
| — | **Macro Average** | **89.43%** | **66.06%** | **73.19%** | **1,137** |
| — | **Weighted Average** | **90.87%** | **90.24%** | **89.99%** | **1,137** |

![Multiclass Performance](assets/multiclass_performance.png)

---

### 3. External Validation on NCBI AMRFinderPlus (11,456 Catalog Entries)

| Metric / Evaluation Subgroup | LightGBM Baseline | ESM-2 Transformer |
| :--- | :---: | :---: |
| **Overall External Sensitivity ($N = 837$)** | 96.65% | **98.33%** |
| **$\beta$-lactam Detection Sensitivity** | 99.85% | **99.85%** |
| **Aminoglycoside Detection Sensitivity** | 72.97% | **89.19%** |
| **External Test Accuracy** | 93.49% | **97.27%** |
| **False Positives on Non-AMR Controls ($N = 195$)** | 26 | **8** |

---

## 📁 Project Structure

```text
Antimicrobial-gene-resistance-AMR-gene-detection/
│
├── app/                                # FastAPI Web Service & Dashboard
│   ├── main.py                         # REST endpoints, sequence validation, inference
│   ├── preset_sequences.json           # 26 curated clinical presets across 9 phenotypes
│   ├── static/                         # Frontend styles, client JS, and assets
│   └── templates/index.html            # Visual Analytics & Classic UI templates
│
├── assets/                             # Figures, architecture diagrams, UI screenshots
│   ├── system_architecture.png         # Pipeline flowchart
│   ├── ui_dashboard_amr_positive.png   # Web app AMR positive detection screenshot
│   ├── ui_dashboard_non_amr_control.png# Web app Non-AMR control screenshot
│   ├── model_benchmark.png             # Performance bar chart
│   ├── multiclass_performance.png      # 9-class precision/recall tradeoff
│   └── xai_attribution.png             # Integrated Gradients residue profile
│
├── configs/
│   └── config.yaml                     # Model hyperparameters & training configurations
│
├── data/
│   ├── processed/                      # Curated zero-leakage splits & feature scalers
│   │   ├── train_split.csv             # 5,841 training sequences
│   │   ├── val_split.csv               # 692 validation sequences
│   │   ├── test_split.csv              # 1,137 held-out test sequences
│   │   ├── positive_sequences_curated.fasta
│   │   ├── negative_sequences_curated.fasta
│   │   ├── kmer_extractor.pkl          # Fitted 4,000-motif TF-IDF vectorizer
│   │   └── biophysical_scaler.pkl      # Fitted 6-property StandardScaler
│   └── external/
│       └── amrfinder/ReferenceGeneCatalog.txt # NCBI AMRFinderPlus reference catalog
│
├── models/                             # Pretrained & fine-tuned model checkpoints
│   ├── lightgbm_model.pkl              # Primary binary LightGBM classifier
│   ├── multiclass_lightgbm.pkl         # 9-class drug family LightGBM classifier
│   ├── baseline_xgboost.pkl            # XGBoost baseline
│   ├── baseline_randomforest.pkl       # Random Forest baseline
│   ├── baseline_logisticregression.pkl # Logistic Regression baseline
│   └── esm2_amr_classifier.pt          # Fine-tuned ESM-2 (35M) weights (Git LFS)
│
├── results/                            # Benchmark reports, JSONs, and evaluation CSVs
│   ├── metrics/all_models_benchmark.csv
│   └── metrics/multiclass_metrics.json
│
├── src/                                # Core bioinformatics & ML modules
│   ├── data/pipeline_data.py           # Ingestion and curation pipelines
│   ├── preprocessing/                  # Sanitization, MinHash clustering, splitting
│   ├── features/                       # k-mer TF-IDF & physicochemical properties
│   ├── models/                         # Baseline training, ESM-2 fine-tuning, ensembling
│   └── explainability/                 # Integrated Gradients & in-silico mutagenesis
│
├── tests/                              # Unit & integration test suite
│   ├── test_sequence_validator.py
│   ├── test_cluster_splitter.py
│   └── test_config.py
│
├── .gitattributes                      # Git LFS configuration for *.pt weights
├── .gitignore                          # Clean repository hygiene rules
├── LICENSE                             # MIT Open-Source License
├── main_pipeline.py                    # Master orchestrator script
└── requirements.txt                    # Python dependencies
```

---

## ⚙ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/R-Sai-Sathvik/Antimicrobial-gene-resistance-AMR-gene-detection.git
cd Antimicrobial-gene-resistance-AMR-gene-detection
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Unit Tests
```bash
python -m pytest tests
```

---

## 🚀 Running the Application

### Launch the FastAPI Server & Web Interface
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

* Explore 26 curated clinical presets across 9 antibiotic phenotypes.
* Input raw FASTA peptide strings or raw nucleotide DNA strings (auto-translated via six-frame translation).
* Inspect model consensus, probability meters, and residue attribution highlights.

---

## 📡 REST API Endpoints

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Serves the interactive Web Analytics Dashboard. |
| `POST` | `/api/v1/predict` | Runs validation, dual-path inference, ensembling, and multi-class prediction on input sequence. |
| `GET` | `/api/v1/health` | Returns server health status and loaded model artifacts. |
| `GET` | `/api/v1/presets` | Retrieves curated clinical presets across all 9 drug classes. |
| `GET` | `/api/v1/benchmark-metrics` | Returns test-set performance metrics across all models. |

### Example Request (`POST /api/v1/predict`)
```json
{
  "sequence": "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW",
  "sequence_id": "blaTEM-1_Sample"
}
```

### Example Response
```json
{
  "sequence_id": "blaTEM-1_Sample",
  "length": 286,
  "is_amr": true,
  "amr_probability": 0.986,
  "confidence_level": "High",
  "predicted_drug_class": "Beta-lactam",
  "predicted_mechanism": "Enzymatic Inactivation",
  "top_contributing_residues": [
    {"position": 70, "residue": "S", "attribution": 0.980, "motif": "STFK"}
  ],
  "biophysical_properties": {
    "length": 286,
    "molecular_weight": 31515.2,
    "isoelectric_point": 5.42,
    "gravy_index": -0.114,
    "aromaticity": 0.087,
    "net_charge_ph7": -4.85
  },
  "esm2_probability": 0.991,
  "warnings": [],
  "model_version": "ConsensusEnsemble_v1.0"
}
```

---

## 📚 Citation & References

If you use this codebase or model architecture, please cite the primary data repositories and foundational works:

1. **CARD:** Alcock et al., *"CARD 2023: expanded curation, support for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance Database,"* *Nucleic Acids Res.*, 51(D1):D690–D699, 2023. [doi:10.1093/nar/gkac920](https://doi.org/10.1093/nar/gkac920).
2. **ESM-2:** Lin et al., *"Evolutionary-scale prediction of atomic-level protein structure with a language model,"* *Science*, 379(6637):1123–1130, 2023. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574).
3. **AMRFinderPlus:** Feldgarden et al., *"Validating the AMRFinder tool and the NCBI bacterial antimicrobial resistance reference gene database,"* *Antimicrob. Agents Chemother.*, 63(11):e00483-19, 2019. [doi:10.1128/AAC.00483-19](https://doi.org/10.1128/AAC.00483-19).
4. **Captum / Integrated Gradients:** Sundararajan et al., *"Axiomatic attribution for deep networks,"* *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, PMLR 70:3319–3328, 2017.
5. **UniProt:** The UniProt Consortium, *"UniProt: the Universal Protein Knowledgebase in 2023,"* *Nucleic Acids Res.*, 51(D1):D523–D531, 2023. [doi:10.1093/nar/gkac1052](https://doi.org/10.1093/nar/gkac1052).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
