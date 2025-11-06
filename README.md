# 🧠 Nano-LCM: Quantum-Conditioned IoT Intrusion Detection

This repository contains **`nanolcm.py`**, a PyTorch + PennyLane implementation of a **quantum-conditioned lightweight transformer (Nano-LCM)** for intrusion detection.  
It is trained on the **UNB/CIC IoT dataset**, which you can **download from Kaggle** and place locally.

---

## ⚙️ Overview
- Builds a **balanced benign vs. attack dataset** from the Kaggle UNB/CIC IoT data  
- Combines **Quantum Hilbert Parameterization (PennyLane)** with a **Tiny Transformer**  
- Uses **Quantum Context Gating (QCG)** for FiLM-style modulation  
- Evaluates using **Accuracy, Precision, Recall, and F1-score**

---

## 📦 Setup Instructions

### 1️⃣ Download Dataset
- Visit the Kaggle dataset page:  
  👉 [UNB CIC IoT Dataset on Kaggle](https://www.kaggle.com/datasets)  
- Download and extract it inside your project directory so it looks like:

your-repo/
├── dataset/
│   ├── wataiData/
│   │   └── csv/
│   │       ├── file1.csv
│   │       └── …
├── nanolcm.py
└── README.md

---

### 2️⃣ Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate


⸻

3️⃣ Install Dependencies

pip install "torch>=2.2" "pennylane>=0.36" "pandas>=2.1" "numpy>=1.26" "tqdm>=4.66"


⸻

▶️ Run the Script
	1.	Open nanolcm.py and set the dataset path:

ROOT = "./dataset/wataiData/csv"


	2.	Run the training script:

python nanolcm.py


	3.	The balanced dataset files will be saved to:

./CICIoT2023_balanced_30k_30k.csv
./CICIoT2023_balanced_30k_30k.parquet



⸻

🧾 Credits
	•	Dataset: UNB/CIC IoT (Kaggle)
	•	Frameworks: PyTorch, PennyLane

