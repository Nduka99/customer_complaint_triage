# Agentic Consumer Complaint Triage and Resolution System

A deep learning pipeline for automated classification, regulatory guidance retrieval, and intelligent routing of consumer financial complaints. Built on 1.8 million real complaints from the Consumer Financial Protection Bureau (CFPB), the system classifies complaints into 10 product categories, retrieves relevant regulatory examination procedures via hybrid RAG, and uses a Thompson Sampling bandit to route complaints optimally — including learning when to escalate to human review.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Technical Approach](#technical-approach)
- [Repository Structure](#repository-structure)
- [Notebook Descriptions](#notebook-descriptions)
- [Data](#data)
- [Models and Training](#models-and-training)
- [Knowledge Distillation](#knowledge-distillation)
- [RAG Pipeline](#rag-pipeline)
- [RL Routing](#rl-routing)
- [Evaluation Summary](#evaluation-summary)
- [Hardware and Reproducibility](#hardware-and-reproducibility)
- [Installation](#installation)
- [Limitations and Future Work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

Financial institutions receive thousands of consumer complaints daily. Manual triage takes approximately 30 minutes per complaint reading the text, identifying the product category, locating relevant regulatory guidance, and determining the appropriate resolution path. This project automates that pipeline using transformer-based classification, retrieval-augmented generation, and reinforcement learning.

The system is engineered to process a complaint in under 3 seconds, handling 92% of cases automatically and flagging the remaining 8% for human review based on learned confidence thresholds.

This project was developed as part of a Masters-level Deep Learning for NLP module. All experiments use real CFPB data with temporal train/test splits, and all evaluation is conducted on held-out test sets containing over 274,000 samples.

---

## Key Results

| Model                        | Method                             | Macro-F1 | vs LogReg Baseline |
| ---------------------------- | ---------------------------------- | -------- | ------------------ |
| Logistic Regression + TF-IDF | Traditional ML (full 1.8M)         | 0.6891   | --                 |
| RoBERTa-base                 | Standard fine-tuning (500K)        | 0.7298   | +0.041             |
| ModernBERT-base              | Standard fine-tuning (500K)        | 0.7360   | +0.047             |
| RoBERTa-base                 | Knowledge distillation (full 1.8M) | 0.7496   | +0.061             |
| LR Stacking Ensemble         | Distilled RoBERTa + ModernBERT     | 0.7574   | +0.068             |

The LogReg baseline scored 0.754 when evaluated on a 100K subsample, which appeared competitive. When retrained and evaluated on the full dataset, its macro-F1 dropped to 0.689, revealing that subsample evaluation inflated the baseline. The deep learning pipeline outperforms the true baseline by nearly 7 points, with each technique contributing measurable improvement.

---

## System Architecture

```
Consumer Complaint (raw text)
         |
         v
+------------------------+
|  Product Classifier    |     RoBERTa-D + ModernBERT + LR Stacking Ensemble
|  (10-class, 0.757 F1)  |     Confidence score computed from max softmax
+----------+-------------+
           |
     +-----+------+
     |             |
     v             v
 conf >= 0.60   conf < 0.60
     |             |
     v             v
+----------+  +-----------+
| Product- |  | Unfiltered|
| filtered |  | RAG or    |
| RAG      |  | Human     |
| (1.00    |  | Escalation|
|  hit@3)  |  |           |
+----------+  +-----------+
     |             |
     v             v
+------------------------+
|  RL Routing Bandit     |     4-arm Thompson Sampling
|  (RoBERTa-D /         |     Trained on 1.8M real
|   ModernBERT /         |     resolution outcomes
|   Ensemble /           |
|   Human Escalation)    |
+----------+-------------+
           |
           v
+------------------------+
|  Agent Response        |     Regulatory context + routing
|  Classification +      |     decision + escalation flag
|  Confidence + Context  |     if applicable
+------------------------+
```

---

## Technical Approach

### Classification

Two transformer architectures with meaningfully different design choices:

| Feature             | RoBERTa-base (125M)        | ModernBERT-base (149M)                   |
| ------------------- | -------------------------- | ---------------------------------------- |
| Positional encoding | Absolute (learned)         | RoPE (rotary)                            |
| Attention pattern   | Full attention, all layers | Alternating full + sliding window        |
| Tokeniser           | BPE (50K vocab)            | Custom BPE (50K vocab, different splits) |
| Pretraining data    | 160 GB                     | 2 TB                                     |
| Context window      | 512 tokens                 | 8,192 tokens                             |

Both models were trained with the same hyperparameters (determined via ablation study on RoBERTa), enabling direct comparison. The architectural differences produce complementary error patterns, which the ensemble exploits.

### Knowledge Distillation

A BART-large-mnli teacher model (400M parameters) generated soft probability distributions for 25,000 training samples via zero-shot NLI classification. The distilled RoBERTa was then trained on the full 1.8M dataset with a hybrid loss:

- 25K samples with teacher labels: alpha * CE(hard) + (1-alpha) * T^2 * KL(soft)
- 1.775M samples without: standard weighted cross-entropy

This improved macro-F1 from 0.730 to 0.750 (+0.020), with particular gains on classes where inter-class confusion was highest.

### Retrieval-Augmented Generation

The RAG pipeline uses 12 real CFPB examination procedure PDFs (mortgage servicing, FDCPA, FCRA, TILA, EFTA, compliance management) chunked into ~400-word passages. Retrieval combines:

- Dense retrieval via sentence-transformers/all-MiniLM-L6-v2 embeddings in ChromaDB
- Sparse retrieval via BM25
- Reciprocal Rank Fusion to combine ranked lists

When the classifier's product prediction is used to filter the knowledge base (the production architecture), Hit Rate@3 reaches 1.000. Unfiltered retrieval achieves 0.352 Hit Rate@3 with MRR of 0.276.

### RL Routing

A 4-arm Thompson Sampling contextual bandit routes each complaint to the optimal processing path:

| Arm | Action               | When Selected                |
| --- | -------------------- | ---------------------------- |
| 0   | Route via RoBERTa-D  | Context-dependent            |
| 1   | Route via ModernBERT | Context-dependent            |
| 2   | Route via Ensemble   | Default best performer       |
| 3   | Escalate to human    | Low confidence or hard class |

The bandit is trained on real CFPB resolution outcomes (1.8M complaints with response labels). Reward is binary: positive resolution = 1.0, negative = 0.0, human escalation = 0.5. The escalation arm ensures the system knows when to defer, processing 92% of complaints automatically while flagging 8% for human review.

---

## Repository Structure

```
cfpb-complaint-triage/
|
|-- README.md
|-- requirements.txt
|
|-- data/
|   |-- processed/
|   |   |-- train.parquet           (1,813,849 rows)
|   |   |-- val.parquet             (331,178 rows)
|   |   |-- test.parquet            (274,065 rows)
|   |   |-- label_encoders.pkl
|   |
|   |-- cfpb_regulatory_docs/       (12 CFPB examination procedure PDFs)
|   |-- vector_store/               (ChromaDB persistent storage)
|
|-- notebooks/
|   |-- 01_text_as_data.ipynb
|   |-- 02_data_acquisition_eda.ipynb
|   |-- 03_preprocessing_pipeline.ipynb
|   |-- 03b_class_separability_analysis.ipynb
|   |-- 04b_modernbert_baseline.ipynb
|   |-- 04c_roberta_baseline.ipynb
|   |-- 05_ablation_study.ipynb
|   |-- 06a_roberta_full_train.ipynb
|   |-- 06b_modernbert_full_train.ipynb
|   |-- 07_teacher_inference.ipynb
|   |-- 08_distilled_roberta.ipynb
|   |-- 08b_ml_baselines.ipynb
|   |-- 09_ensemble.ipynb
|   |-- 10_rag_pipeline.ipynb
|   |-- 11_rl_routing.ipynb
|
|-- models/
|   |-- roberta_full/               (non-distilled RoBERTa weights + predictions)
|   |-- modernbert_full/            (ModernBERT weights + predictions)
|   |-- roberta_distilled/          (distilled RoBERTa weights + predictions)
|   |-- ensemble/                   (ensemble predictions + results)
|   |-- ablation_study/             (ablation configs + results)
|   |-- distillation/               (teacher soft labels)
|   |-- ml_baselines/               (LogReg predictions)
|   |-- rag_pipeline/               (knowledge base + evaluation)
|   |-- rl_routing/                 (bandit state + experiment results)
|
|-- reports/
|   |-- figures/                    (all visualisations by notebook)
```

---

## Notebook Descriptions

| Notebook | Purpose                     | Key Output                                                                                        |
| -------- | --------------------------- | ------------------------------------------------------------------------------------------------- |
| 01       | Text as Data                | Tokenisation comparison (BPE variants), UMAP embeddings, attention heatmaps                       |
| 02       | Data Acquisition and EDA    | CFPB dataset exploration, class distributions, volume analysis                                    |
| 03       | Preprocessing Pipeline      | Temporal train/val/test split, label encoding, metadata features, parquet outputs                 |
| 03b      | Class Separability Analysis | 7-method separability analysis, confusion patterns, hard class identification                     |
| 04b/04c  | Subsample Baselines         | 100K-sample 3-epoch runs establishing baseline performance for ModernBERT and RoBERTa             |
| 05       | Ablation Study              | 5 single-variable experiments on RoBERTa identifying max_len=384 as the only positive contributor |
| 06a      | RoBERTa Full Training       | Full-data training with pre-tokenisation and dynamic padding. Epoch scouting for ModernBERT       |
| 06b      | ModernBERT Full Training    | Epoch-capped training informed by RoBERTa scouting, saving 20+ hours of GPU time                  |
| 07       | Teacher Inference           | BART-large-mnli zero-shot NLI producing 25K soft probability distributions                        |
| 08       | Distilled RoBERTa           | Hybrid loss training on full 1.8M with teacher soft labels on 25K subset                          |
| 08b      | ML Baselines (Full Data)    | LogReg on full 1.8M proving subsample evaluation was misleading (0.754 to 0.689)                  |
| 09       | Ensemble                    | 5 ensemble strategies tested. LR stacking achieves 0.757, the best overall result                 |
| 10       | RAG Pipeline                | Real CFPB regulatory documents, hybrid BM25+dense retrieval, product-filtered evaluation          |
| 11       | RL Routing                  | 4-arm Thompson Sampling bandit, 4 experiments, agentic chain demo, RLHF theory                    |
| 12       | Comprehensive Evaluation    | Classification metrics, McNemar's test, bootstrap CIs, temporal drift, and IG explainability      |

---

## Data

The CFPB Consumer Complaint Database contains consumer complaints about financial products and services submitted to the CFPB. Each complaint includes the consumer's narrative description, the product category, the specific issue, and the company's response.

**10 product classes** (with extreme imbalance):

| Class                | Train Samples | Proportion |
| -------------------- | ------------- | ---------- |
| Credit Report        | 956,601       | 52.7%      |
| Debt Collection      | 266,081       | 14.7%      |
| Credit Card          | 168,930       | 9.3%       |
| Bank Account         | 129,869       | 7.2%       |
| Mortgage             | 124,631       | 6.9%       |
| Money Transfer       | 48,354        | 2.7%       |
| Student Loan         | 47,768        | 2.6%       |
| Vehicle Loan         | 35,188        | 1.9%       |
| Payday/Personal Loan | 34,581        | 1.9%       |
| Debt Management      | 1,846         | 0.1%       |

The temporal split uses pre-2025 data for training and mid-2025 onwards for testing, simulating real-world deployment conditions.

---

## Models and Training

### Training Infrastructure

All training was conducted on a single NVIDIA RTX 4060 (8 GB VRAM) with 64 GB system RAM. Key engineering decisions driven by hardware constraints:

- **Pre-tokenisation**: Tokenise the full dataset once upfront, eliminating millions of redundant tokeniser calls per epoch. Reduced estimated training time from 97 hours to 27 hours for the same configuration.
- **Dynamic padding**: DataCollatorWithPadding pads each batch to the longest sequence in that batch rather than to the maximum length. Since median complaint length is approximately 184 tokens, most batches pad to roughly 200 instead of 384.
- **Epoch scouting**: RoBERTa (faster model) trains first to identify the optimal epoch. ModernBERT's max_epochs is then capped at RoBERTa's best epoch + 1, saving approximately 20 hours of GPU time.
- **Stratified subsampling**: When full-data training was infeasible (e.g., initial 1.8M attempt at 97 hours), 500K stratified subsamples were used, providing 5x the baseline data while remaining within a single overnight training run.

### Ablation Study

Each technique was tested independently against the baseline, changing exactly one hyperparameter per run:

| Ablation             | Change                                   | Macro-F1 | Delta  |
| -------------------- | ---------------------------------------- | -------- | ------ |
| Baseline             | CE + class weights, lr=2e-5, max_len=256 | 0.6681   | --     |
| Focal Loss (gamma=2) | Replace CE                               | 0.6648   | -0.003 |
| max_len=384          | Increase context                         | 0.6694   | +0.001 |
| LLRD (decay=0.95)    | Layer-wise LR decay                      | 0.6644   | -0.004 |
| Warmup 6%            | Reduce from 10%                          | 0.6670   | -0.001 |
| Combined best        | max_len=384 only                         | 0.6726   | +0.005 |

Only max_len=384 showed positive contribution. This was applied to both models for full-data training.

---

## Knowledge Distillation

### Motivation

Standard fine-tuning produced severe confusion on hard classes. Debt Management complaints were misclassified 48.5% of the time (28.7% to Debt Collection, 19.8% to Credit Report). With only 1,846 training samples, the model lacked signal to learn class boundaries.

### Teacher Model Selection

Initial experiments with Qwen2.5-7B-Instruct (4-bit quantised) produced degenerate distributions — the model collapsed to a single class for most inputs. The digit-logit extraction approach also failed due to the model generating preamble text before digits.

The final teacher was facebook/bart-large-mnli (400M parameters), a model specifically trained for natural language inference. It produces calibrated probability distributions by computing entailment between the complaint text and each class hypothesis. This is architecturally principled for zero-shot classification rather than being a workaround applied to a generative model.

### Hybrid Loss

The distilled model trains on the full 1.8M dataset. Only the 25K samples with teacher soft labels receive the distillation loss component. The remaining 1.775M samples use standard weighted cross-entropy. This provides full-data volume (all 1,846 Debt Management samples participate in training) while the teacher's soft labels teach inter-class similarity on a representative subset.

---

## RAG Pipeline

### Knowledge Base

12 real CFPB examination procedure PDFs were downloaded from consumerfinance.gov, covering:

- Mortgage Servicing (RESPA, Regulation X, Regulation Z)
- Debt Collection (FDCPA, Regulation F)
- Credit Reporting (FCRA)
- Credit Cards (TILA, Credit CARD Act)
- Deposit Accounts (EFTA, Regulation E)
- Student Lending (TILA Subpart E)
- Auto Lending (TILA, ECOA)
- Remittance Transfers (EFTA Subpart B)
- Payday/Short-term Lending (TILA)
- Compliance Management (CMS Review)

Documents were extracted with PyMuPDF, chunked into approximately 400-word passages with 50-word overlap, and tagged with product and issue metadata via keyword matching.

### Retrieval Evaluation

| Configuration                              | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR   |
| ------------------------------------------ | ---------- | ---------- | ---------- | ----- |
| Unfiltered (full knowledge base)           | 0.187      | 0.352      | 0.450      | 0.276 |
| Product-filtered (production architecture) | 1.000      | 1.000      | 1.000      | 1.000 |

Product-filtered retrieval uses the classifier's predicted product to scope the search to only that product's regulatory documents. This is the intended production architecture — the classifier and RAG pipeline are designed to work together. When classifier confidence falls below 0.60, the system falls back to unfiltered retrieval or human escalation.

---

## RL Routing

### Bandit Design

The Thompson Sampling contextual bandit maintains Beta distribution posteriors per (context, arm) pair, where context is the predicted product class (10 categories) and arms are the 4 routing options. At each step, the bandit samples from each arm's posterior and selects the arm with the highest sample. After observing the outcome, the winning arm's posterior is updated.

The escalation arm receives a fixed reward of 0.5, meaning the bandit only escalates when it estimates automated arms have less than 50% chance of positive resolution. This threshold emerges naturally from the Bayesian update rather than being hardcoded.

### Experiment Results

| Strategy                  | Avg Reward | Escalation Rate |
| ------------------------- | ---------- | --------------- |
| Random routing            | 0.859      | N/A             |
| Static ensemble           | 0.876      | N/A             |
| Thompson Sampling (3-arm) | 0.873      | N/A             |
| Thompson Sampling (4-arm) | 0.875      | 8.0%            |
| Oracle (hindsight)        | 0.951      | N/A             |

The high baseline reward (0.859 for random) reflects that most CFPB complaints receive positive resolutions regardless of routing. The bandit's primary contribution is identifying the 8% of complaints that need human intervention — the hard cases where automated classification is unreliable.

---

## Evaluation Summary

### Progressive Improvement

```
LogReg on 100K subsample (misleading)    0.754
LogReg on full 1.8M (true baseline)      0.689  (subsample was not representative)
RoBERTa non-distilled                    0.730  (+0.041 over true baseline)
ModernBERT non-distilled                 0.736  (+0.047)
RoBERTa distilled                        0.750  (+0.061)
LR Stacking Ensemble                     0.757  (+0.068)
```

### Hard Class Analysis & Systematic Confusions

While the overall misclassification rate is 13.8% (37,745 errors out of 274,065 test samples), these errors are highly systematic and driven by genuine category overlap rather than random model failure. 

**Top Confusion Pairs (39.5% of all errors):**
- `Debt Collection` → `Credit Report` (27.9% of errors)
- `Credit Report` → `Debt Collection` (11.6% of errors)
- `Credit Card` → `Credit Report` (9.1% of errors)

The model often makes these errors with very high confidence (>0.99) because consumers consistently mention the impact on their credit reports when complaining about debt collection or credit cards. This creates legitimate linguistic and regulatory ambiguity.

Furthermore, Debt Management (0.1% of training data, 1,846 samples) remains the most challenging individual class with approximately 35% F1. Its primary error modes (`Debt Mgmt` → `Debt Collection` at 28.7%, `Debt Mgmt` → `Credit Report` at 19.8%) perfectly mirror the broader dataset trends.

The system addresses these overlapping categories through the confidence-based escalation mechanism: ambiguous complaints with confidence below 0.60 (which captures many of these overlapping edge cases) are routed to human review rather than processed automatically.

---

## Hardware and Reproducibility

**Hardware**: NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), 64 GB DDR5 RAM, AMD Ryzen 7000, Windows 11.

**Reproducibility**: All experiments use seed=42. Stratified sampling uses the same seed throughout. Pre-tokenisation ensures identical data across runs. Training arguments and model configurations are saved as JSON for each experiment.

**Approximate compute budget**:

| Component                                   | Time                |
| ------------------------------------------- | ------------------- |
| Ablation study (5 runs, 100K, 1 epoch each) | 2.5 hours           |
| RoBERTa full training (500K, 6 epochs)      | 15 hours            |
| ModernBERT full training (500K, 4 epochs)   | 22 hours            |
| Teacher inference (BART-large-mnli, 25K)    | 4 hours             |
| Distilled RoBERTa (1.8M, 7 epochs)          | 40 hours            |
| ML baselines, ensemble, RAG, RL             | 3 hours             |
| **Total**                             | **~87 hours** |

---

## Installation

```bash
git clone https://github.com/[username]/cfpb-complaint-triage.git
cd cfpb-complaint-triage
pip install -r requirements.txt
```

**Core dependencies**:

```
torch>=2.2.0
transformers>=4.40.0
datasets>=2.19.0
accelerate>=0.30.0
sentence-transformers>=3.0.0
chromadb>=0.5.0
rank-bm25>=0.2.2
scikit-learn>=1.4.0
xgboost>=2.0.0
pymupdf>=1.24.0
numpy>=1.26.0
pandas>=2.2.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

**Note**: The CFPB complaint dataset is publicly available at https://www.consumerfinance.gov/data-research/consumer-complaints/. The preprocessing pipeline (NB03) handles download and temporal splitting. Model weights are not included in this repository due to file size; they can be reproduced by running the training notebooks.

---

## Limitations and Future Work

**Current limitations**:

- Debt Management classification (35% F1) reflects genuine category overlap in the CFPB taxonomy rather than model failure, but remains an unsolved challenge.
- Knowledge distillation used BART-large-mnli (400M) as teacher. A larger teacher model (7B+) would likely produce richer soft distributions but was infeasible on the available hardware.
- The RAG knowledge base uses CFPB examination procedures as a proxy for complete regulatory guidance. A production system would incorporate consent orders, supervisory highlights, and company-specific policies.
- The RL bandit operates on historical data. Online learning in a production deployment would require careful monitoring for reward signal drift.

**Future directions**:

- DeBERTa-v3 integration: the original primary model failed during training due to a known compatibility issue. Resolving this would provide a third architecture with genuinely different attention (disentangled).
- Multi-task learning: joint training on product classification, issue identification, and response prediction using shared representations.
- Hierarchical classification: a two-stage classifier that first groups similar categories (e.g., Debt Management + Debt Collection) then distinguishes within the group.
- Online bandit deployment with production monitoring and reward signal validation.
- ONNX export and INT8 quantisation for production-grade latency (planned as NB13).

---

## Acknowledgements

This project was developed as coursework for the Deep Learning for NLP module at Masters level. The CFPB Consumer Complaint Database is maintained by the Consumer Financial Protection Bureau and is publicly available. All regulatory documents used in the RAG pipeline are publicly published examination procedures from consumerfinance.gov.
