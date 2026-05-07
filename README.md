# SCOPE: Siamese Contrastive Operon Pair Embeddings

**SCOPE** is a project evaluating Siamese neural network architectures for operonic pair classification — the binary task of predicting whether two consecutive proteins belong to the same transcription unit (operon). We compare embedding-based models using pre-trained protein language models against traditional machine learning baselines leveraging hand-crafted physicochemical features.

---

## Repository Structure

```
├── operon-pair-classification-logistic-regression.ipynb   # Logistic Regression and XGBoost baselines
├── esm-2-3b-siamese-nn-for-operon-pair-classification.ipynb  # Siamese MLP with ESM-2 3B encoder
├── esm-2-protbert.ipynb                                   # Siamese MLP with ProtBERT-BFD encoder
```

---

## Dataset

The training and validation data are hosted on Kaggle. You will need a Kaggle account to download them.

**Dataset:** [Operon Pair Classification v2](https://www.kaggle.com/datasets/kennethrodrigues/operon-pair-classification-v2)

The dataset consists of pairs of amino acid sequences with a binary label indicating whether the two proteins are co-operonic (`1`) or not (`0`). The data is sourced from ODB (Operon DataBase) across *E. coli*, *Vibrio cholerae*, and *Synechococcus elongatus* genomes.

To use the dataset on Kaggle:
1. Go to the dataset page linked above.
2. Click **Download** or add it directly to your Kaggle notebook via **Add Data**.
3. The files `operon_train.csv` and `operon_val.csv` will be available at `/kaggle/input/datasets/kennethrodrigues/operon-pair-classification-v2/`.

Final evaluation is performed on the DGEB benchmark datasets, available via HuggingFace:
```python
cyano  = pd.read_parquet("hf://datasets/tattabio/cyano_operonic_pair/data/train-00000-of-00001.parquet")
vibrio = pd.read_parquet("hf://datasets/tattabio/vibrio_operonic_pair/data/train-00000-of-00001.parquet")
ecoli  = pd.read_parquet("hf://datasets/tattabio/ecoli_operonic_pair/data/train-00000-of-00001.parquet")
```

---

## Approaches

### 1. Logistic Regression and XGBoost Baselines
**Notebook:** `operon-pair-classification-logistic-regression.ipynb`

Both baseline models operate on hand-crafted physicochemical features extracted from amino acid sequences. Each protein sequence is converted into a 61-dimensional feature vector capturing:
- Amino acid frequencies (20 features)
- Biochemical group frequencies (5 features)
- Physicochemical statistics: hydrophobicity, net charge, molecular weight, aromaticity, aliphatic index (7 features)
- Sequence complexity: log length, Shannon entropy, max run fraction, unique AA fraction (4 features)
- Group-level bigrams (25 features)

Each protein pair is then represented as a 305-dimensional vector via a Siamese interaction pattern: `[u, v, u−v, |u−v|, u⊙v]`.

**Logistic Regression** is trained using `scikit-learn`'s `LogisticRegression` with `C=1.0`, L2 penalty, `class_weight="balanced"`, and the `lbfgs` solver. Features are standardized with `StandardScaler`.

**XGBoost** is trained using `XGBClassifier` with 300 estimators, max depth 6, learning rate 0.05, and `scale_pos_weight` set to the negative-to-positive class ratio. Early stopping with patience 20 is applied on validation AUC.

| Model | Average Precision | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.41 | 0.63 |
| XGBoost | 0.40 | 0.62 |

---

### 2. Siamese MLP with ESM-2 3B
**Notebook:** `esm-2-3b-siamese-nn-for-operon-pair-classification.ipynb`

Each protein sequence is encoded using [ESM-2 3B](https://huggingface.co/facebook/esm2_t36_3B_UR50D) (3 billion parameters, 2560-dimensional embeddings), loaded via the Hugging Face `transformers` library. Sequences are mean-pooled across token positions to yield a fixed-size embedding.

The two embeddings are fused via concatenation, signed difference, absolute difference, and element-wise product, producing a 10,240-dimensional fused vector. This is passed to an MLP classifier with hidden layers `[4096, 1024, 256, 64]`, trained with:
- **Optimizer:** AdamW, lr=1e-4, weight decay=0.1
- **Loss:** BCEWithLogitsLoss with `pos_weight` for class imbalance
- **Regularization:** Dropout (0.5), BatchNorm, label smoothing (0.1), gradient clipping (norm=1.0)
- **Scheduler:** Cosine annealing
- **Early stopping:** Patience 20 on validation AUROC

| Model | Average Precision | ROC-AUC |
|---|---|---|
| ESM-2 3B + MLP | 0.45 | 0.71 |

---

### 3. Siamese MLP with ProtBERT-BFD
**Notebook:** `esm-2-protbert.ipynb`

Each protein sequence is encoded using [ProtBERT-BFD](https://huggingface.co/Rostlab/prot_bert_bfd) (420M parameters, 1024-dimensional embeddings), loaded via the Hugging Face `transformers` library. The same Siamese fusion strategy is applied, producing a 5,120-dimensional fused vector. This is passed to an MLP classifier with hidden layers `[2048, 512, 128, 32]`, trained with:
- **Optimizer:** AdamW, lr=1e-4, weight decay=1e-2
- **Loss:** BCEWithLogitsLoss with `pos_weight` for class imbalance
- **Regularization:** Dropout (0.3), BatchNorm, label smoothing (0.1)
- **Scheduler:** Cosine annealing
- **Early stopping:** On validation AUROC

| Model | Average Precision | ROC-AUC |
|---|---|---|
| ProtBERT-BFD + MLP | 0.51 | 0.71 |

---

## Computational Environment

All experiments were run on **Kaggle** using a single **NVIDIA T4 GPU (16GB)**. Embedding generation and MLP training for the ESM-2 3B model used a batch size of 4 for encoding and 256 for MLP training. ProtBERT-BFD used a batch size of 4 for encoding and 128 for MLP training.

---

## Results Summary

| Model | Average Precision | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.41 | 0.63 |
| XGBoost | 0.40 | 0.62 |
| ESM-2 3B + MLP | 0.45 | 0.71 |
| ProtBERT-BFD + MLP | 0.51 | 0.71 |
| DGEB Baseline (cosine similarity) | 0.52 | — |

---

## Authors

- Akarsh Gupta
- Kenneth Rodrigues
- Sagnik Chatterjee

*COMPSCI 690U @ UMass Amherst, 2026*