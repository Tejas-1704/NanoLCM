# 🧠 Nano-LCM: Quantum-Conditioned IoT Intrusion Detection

This repository contains **`nanolcm.py`**, a PyTorch + PennyLane implementation of a **quantum-conditioned lightweight transformer (Nano-LCM)** for intrusion detection.
Train it locally using the **UNB/CIC IoT dataset** that you **download from Kaggle**.

---

## ⚙️ Overview

* Builds a **balanced benign vs. attack** dataset from the Kaggle UNB/CIC IoT CSVs
* **Quantum Hilbert Parameterization (PennyLane)** + **Tiny Transformer** backbone
* **Quantum Context Gating (QCG)** for FiLM-style modulation of token features
* Reports **Accuracy, Precision, Recall, F1-score** each epoch

---

## 📦 Setup (Local)

### 1) Download the dataset

* Download **UNB/CIC IoT dataset** from Kaggle and extract it locally (e.g., `./dataset/wataiData/csv`).

### 2) Create & activate a virtual environment

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install "torch>=2.2" "pennylane>=0.36" "pandas>=2.1" "numpy>=1.26" "tqdm>=4.66"
```

---

## ▶️ Run

1. Set the dataset path inside **`nanolcm.py`**:

```python
ROOT = "./dataset/wataiData/csv"
```

2. Launch training:

```bash
python nanolcm.py
```

---

## 📈 Outputs

Balanced datasets and logs will be written to:

```
./CICIoT2023_balanced_30k_30k.csv
./CICIoT2023_balanced_30k_30k.parquet
```

Example console metrics:

```
Epoch 3: loss=0.2942  acc=0.962  f1=0.958  prec=0.961  rec=0.954
=== Final (val) ===
acc: 0.9621
f1: 0.9583
precision: 0.9610
recall: 0.9540
```

---

## 🔧 Key Parameters (edit in `nanolcm.py`)

```python
# data
ROOT = "./dataset/wataiData/csv"
PER_CLASS = 1000
VAL_RATIO = 0.2
SEED = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# training
EPOCHS = 3
BS = 512
LR = 2e-3
WD = 2e-4

# model (typical defaults)
# d_model=96, n_tokens=8, n_qubits=6, n_layers=3, head_dim=8, hpp_dim=16, tf_layers=1
```

---

## 🧩 Components (high level)

* **QHilbertParam:** PennyLane circuit (RX/RY/RZ + ring CNOT); angle-encoded inputs → expectation-value embedding
* **HilbertParamPool:** Multi-head quantum pooling + projection + LayerNorm
* **QuantumContextGate (QCG):** FiLM-style scale/shift applied to transformer tokens
* **TinyTransformerBlock:** Lightweight attention + MLP with residuals
* **Binary Head:** Single-logit output (benign=0, attack=1)

---

## 🧾 Credits 

* **Dataset:** UNB/CIC IoT (Kaggle)
* **Frameworks:** PyTorch, PennyLane
  
