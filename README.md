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
- [Pipeline Architecture](#-pipeline-architecture)
- [Curated Benchmark Dataset](#-curated-benchmark-dataset)
- [Benchmark Results](#-benchmark-results)
- [Explainable AI & Active-Site Localization](#-explainable-ai--active-site-localization)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [REST API Endpoints](#-rest-api-endpoints)
- [Citation & References](#-citation--references)
- [License](#-license)

---

## 🔬 Overview

Antimicrobial Resistance is responsible for over 1.27 million direct fatalities annually. Conventional phenotypic susceptibility testing (AST) requires 48 to 72 hours of microbial incubation, delaying targeted therapeutic intervention. In addition, standard alignment algorithms (such as BLAST and MMseqs2) suffer acute sensitivity drops on divergent, novel, or rapidly evolving resistance alleles sharing $<60\%$ sequence identity with cataloged references.

This system provides an end-to-end alignment-free diagnostic framework:
1. **Dual-Path Representation:** Extracts 4,000 $k$-mer TF-IDF motifs ($k \in \{3, 4\}$) and 6 biophysical descriptors (GRAVY hydropathy, aromaticity, isoelectric point $pI$, molecular weight $M_W$, net physiological charge at pH 7, and length).
2. **Transfer Learning via ESM-2:** Fine-tunes the 35M-parameter Evolutionary Scale Modeling protein language model (`facebook/esm2_t12_35M_UR50D`) with a specialized LayerNorm classification head.
3. **Calibrated Consensus Ensemble:** Merges gradient-boosted tree patterns with evolutionary transformer embeddings via validation-optimized soft voting ($0.70 \times \text{ESM-2} + 0.30 \times \text{LightGBM}$) with Bayesian confidence damping.
4. **9-Class Phenotypic Subtyping:** Classifies detected determinants into 9 clinical antibiotic phenotypes with cost-sensitive inverse-frequency class weighting.
5. **Axiomatic Explainable AI:** Leverages Integrated Gradients (50 steps) and in-silico alanine scanning to map predictions to experimentally validated catalytic pockets (such as Ser70 nucleophiles in $\beta$-lactamases).
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

## 🏗 Pipeline Architecture

The end-to-end workflow enforces strict sequence hygiene, 70% MinHash homology partition, dual-branch feature engineering, parallel model inference, consensus ensembling, and axiomatic explainability:

![Pipeline Architecture](assets/system_architecture.png)

---

## 📊 Curated Benchmark Dataset

The non-redundant benchmark comprises **7,670 bacterial sequences**:
* **Positive AMR Determinants ($N = 5,337$):** Curated from the Comprehensive Antibiotic Resistance Database (CARD v3.2.8) across enzymatic inactivation, target protection, and active efflux mechanisms annotated via the Antibiotic Resistance Ontology (ARO).
* **Negative Housekeeping Controls ($N = 2,333$):** Curated from UniProtKB/Swiss-Prot across essential cellular proteins (GAPDH, citrate synthase, DNA polymerases, ribosomal proteins) devoid of resistance functions.
* **Leakage-Free Partitioning:** Sequences sharing $\ge 70\%$ identity were grouped into disjoint MinHash clusters. Whole clusters were split into **70% Train** ($N = 5,841$), **15% Validation** ($N = 692$), and **15% Test** ($N = 1,137$) to guarantee zero homology data leakage between splits.

| Dataset Split | AMR Positives | Non-AMR Controls | Total Sequences | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set** | 4,064 | 1,777 | 5,841 | 76.15% |
| **Validation Set** | 482 | 210 | 692 | 9.02% |
| **Test Set (Held-Out)** | 899 | 238 | 1,137 | 14.82% |
| **Total** | **5,337** | **2,333** | **7,670** | **100.0%** |

---

## 📈 Benchmark Results

### 1. Binary Classification Benchmark (Held-Out Test Set, $N = 1,137$)

All models were evaluated on the cluster-isolated test partition under identical preprocessing:

| Architecture | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression ($L_2$)** | 92.98% | 96.23% | 94.87% | 85.71% | 95.55% | 97.03% |
| **XGBoost (Hist Gradient)** | 94.54% | 96.92% | 96.18% | 88.24% | 96.55% | 98.84% |
| **Random Forest (200 Trees)** | 95.23% | 97.78% | 96.18% | 91.60% | 96.97% | 99.02% |
| **LightGBM (Best ML Baseline)** | 95.67% | 97.79% | 96.72% | 91.60% | 97.26% | 98.96% |
| **ESM-2 (35M Transformer)** | 97.27% | 99.14% | 97.56% | 95.90% | 98.34% | 99.62% |
| **Consensus Ensemble** | **97.80%** | **99.25%** | **98.09%** | **96.41%** | **98.67%** | **99.75%** |

![Model Benchmark Comparison](assets/model_benchmark.png)

* **Error Reduction:** Ensembling LightGBM motifs with contextual ESM-2 embeddings reduced false positives by **65%** (from 20 down to 7) and false negatives by **40%** (from 30 down to 18) across 1,137 test sequences.

---

### 2. Multi-Class Drug Resistance Subtyping (9 Classes)

A multi-class LightGBM classifier with inverse-frequency class weighting categorizes confirmed resistance genes into 9 phenotypic classes:

| Class Index | Resistance Category | Precision | Recall | F1-Score | Test Support |
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

Generalization was benchmarked against the independent NCBI AMRFinderPlus Reference Gene Catalog:
* **Overlapping AMR Genes:** 837 confirmed resistance determinants.
* **ESM-2 Sensitivity:** **98.33%** (823/837 detected).
* **LightGBM Sensitivity:** **96.65%** (809/837 detected).
* **Control Specificity ($N = 195$):** ESM-2 produced only 8 false positives (95.90% specificity).

---

## 🔍 Explainable AI & Active-Site Localization

To ensure biological validity rather than reliance on sequence artifacts, we applied **Integrated Gradients (Captum)** across 50 Riemann approximation steps.

In Class A $\beta$-lactamase `blaTEM-1`, the primary attribution peak mapped directly to **residue position 70 inside the conserved `STFK` motif** (attribution score: 0.98), precisely isolating the catalytic Serine nucleophile responsible for $\beta$-lactam ring hydrolysis:

![Explainable AI Residue Attribution Profile](assets/xai_attribution.png)

* **In-Silico Alanine Scanning:** Mutating Ser70 $\to$ Ala70 collapsed predicted resistance probability from **98.60% down to 21.40% ($\Delta P = 77.20\%$)**, whereas mutating neutral residues caused negligible change ($\Delta P < 1.20\%$).

---

## 📁 Project Structure

```text
AMR-Sequence-Intelligence/
│
├── app/                                # FastAPI application & Web Dashboard
│   ├── main.py                         # REST endpoints & model serving logic
│   ├── static/                         # Frontend CSS, JS, and dashboard assets
│   └── templates/index.html            # Interactive visual analytics UI
│
├── assets/                             # Architecture diagrams & UI screenshots
│   ├── system_architecture.png
│   ├── ui_dashboard_amr_positive.png
│   ├── ui_dashboard_non_amr_control.png
│   ├── model_benchmark.png
│   ├── multiclass_performance.png
│   └── xai_attribution.png
│
├── configs/
│   └── config.yaml                     # Central pipeline hyperparameters
│
├── data/
│   ├── processed/                      # Curated zero-leakage splits & fastas
│   │   ├── train_split.csv             # 5,841 training sequences
│   │   ├── val_split.csv               # 692 validation sequences
│   │   ├── test_split.csv              # 1,137 test sequences
│   │   ├── positive_sequences_curated.fasta
│   │   ├── negative_sequences_curated.fasta
│   │   ├── kmer_extractor.pkl          # Fitted 4,000-motif TF-IDF vectorizer
│   │   └── biophysical_scaler.pkl      # Fitted 6-property StandardScaler
│   └── external/                       # NCBI AMRFinderPlus catalog
│       └── amrfinder/ReferenceGeneCatalog.txt
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
├── src/                                # Core library modules
│   ├── data/pipeline_data.py           # Ingestion from CARD and UniProt
│   ├── preprocessing/                  # Sanitization, MinHash clustering, splitting
│   ├── features/                       # k-mer TF-IDF & biophysical descriptors
│   ├── models/                         # Model architectures, training & ensembling
│   └── explainability/                 # Integrated Gradients & in-silico mutagenesis
│
├── tests/                              # Unit & integration tests
│   ├── test_sequence_validator.py
│   ├── test_cluster_splitter.py
│   └── test_config.py
│
├── .gitattributes                      # Git LFS configuration for large weights (*.pt)
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
