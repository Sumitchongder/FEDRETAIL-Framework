# 🏬 FEDRETAIL

## Federated Retail Data Analysis & Learning Framework

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Industry--Grade-success)]()
[![Research](https://img.shields.io/badge/Research-Peer--Reviewed-important)](https://www.ijarsct.co.in/A16909.pdf)
[![License](https://img.shields.io/badge/License-Academic--Use-red)]()

> **An industry-grade, research-driven framework for privacy-preserving centralized, horizontal, and vertical federated learning in retail and e-commerce systems.**

---

## 📌 Overview

**FEDRETAIL (FEDerated REtail daTA analysis and Learning)** is a **modular, reproducible, and industry-ready federated learning framework** designed to enable **collaborative retail analytics without sharing raw data**.

The framework addresses critical challenges in modern retail and E-commerce 5.0 ecosystems, including:

* Data silos across organizations
* Strict privacy regulations (GDPR, PDPA, CCPA)
* Heterogeneous retailer data distributions
* Scalability of distributed machine learning

FEDRETAIL demonstrates that **federated retail intelligence can achieve near-centralized performance while preserving full data locality**.

---

## 🎯 Key Objectives

* 📊 Benchmark federated aggregation strategies (**FedAvg, FSVRG, CO-OP**)
* 🏬 Model realistic **retailer-level heterogeneity**
* 🔐 Preserve **data privacy and locality**
* 🔁 Enable **fully reproducible experimentation**
* 🧩 Provide **clean, modular abstractions** for FL research and prototyping

---

## 🧠 Core Capabilities

### 🔹 Learning Paradigms

* **Centralized Learning** (Baseline)
* **Horizontal Federated Learning (HFL)**
* **Vertical Federated Learning (VFL)**

### 🔹 Aggregation Algorithms

* **FedAvg** – Standard federated averaging
* **FSVRG** – Variance-reduced federated optimization
* **CO-OP** – Cooperative decentralized learning

### 🔹 Data Distributions

* **IID** – Uniform client distributions
* **Non-IID** – Label-skewed, shard-based distributions

### 🔹 Experimental Controls

* Client participation probability
* Communication rounds
* Local training epochs
* Retailer heterogeneity simulation

---

## 🏗️ System Architecture

> **Figure:** *High-level architecture of the FEDRETAIL framework showing data partitioning, local retailer training, secure aggregation, and global model redistribution.*

<p align="center">
<img width="700" height="500" alt="Image" src="https://github.com/user-attachments/assets/2877379f-1634-4601-936f-32e42d792c26" />
</p>

### Design Principles

* Clear separation of concerns
* Stateless, pluggable aggregation algorithms
* Reproducible experiment pipelines
* Extendable to CPS, IoT, and edge deployments

---

## 🔄 Federated Training Workflow

> **Figure:** *End-to-end federated training workflow illustrating client participation, local updates, and iterative global aggregation.*

<p align="center">
<img width="700" height="500" alt="Image" src="https://github.com/user-attachments/assets/8e5496c1-8978-4edf-925c-b21dc4bda4f0" />
</p>

**Workflow**

1. Retailers join the federation
2. Data preprocessing occurs locally
3. Local model training at each retailer
4. Secure transmission of model updates
5. Central aggregation (FedAvg / FSVRG / CO-OP)
6. Global model redistribution
7. Iterative convergence

---

## 📂 Repository Structure

```text
FEDRETAIL/
├── src/
│   ├── algorithms/        # FedAvg, FSVRG, CO-OP
│   ├── models/            # Neural Network & Logistic Regression
│   ├── data/              # IID & Non-IID partitioning
│   ├── training/          # Centralized, HFL, VFL pipelines
│   └── utils/             # Metrics, visualization, security
│
├── experiments/           # Reproducible experiments
│   ├── algorithm_comparison.py
│   ├── hfl_experiments.py
│   └── vfl_experiments.py
│
├── docs/
│   ├── architecture/      # Diagrams (PNG/SVG)
│   ├── results/           # Experimental plots
│   └── paper/             # Manuscript & supplementary material
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── .gitignore
```

---

## 📊 Dataset

### Fashion-MNIST Benchmark

| Property   | Value                        |
| ---------- | ---------------------------- |
| Images     | 70,000 grayscale             |
| Resolution | 28 × 28                      |
| Classes    | 10 retail fashion categories |
| Training   | 60,000                       |
| Testing    | 10,000                       |

Supports both **IID** and **Non-IID** splits to emulate real-world retail data heterogeneity.

---

Absolutely, Sumit. Here's a refined and recruiter-ready **📈 Experimental Results** section for your README, directly based on the five uploaded figures. It emphasizes clarity, technical depth, and federated learning insights across architectures and participation settings:

---

## 📈 Experimental Results

### 🧠 Federated Algorithm Performance 

> 📌 **Algorithm Comparison Plot**  
> **Caption:** *Accuracy and loss comparison of FedAvg, FSVRG, and CO-OP under IID data distribution.*

<p align="center">
<img width="700" height="300" alt="Image" src="https://github.com/user-attachments/assets/3fddfb52-d50f-43d3-bfcd-b6ff0221fa15" />
</p>

- **FedAvg** consistently achieves the highest accuracy (~0.95) and lowest loss across 100 epochs.
- **FSVRG** improves gradient consistency but converges slower than FedAvg.
- **CO-OP** reduces communication overhead, though with trade-offs in convergence speed and final accuracy.

---

### 🔄 HRFL with Two-Layer Neural Network 

> 📌 **HRFL Performance Plot**  
> **Caption:** *Accuracy and loss comparison across retailers using Hierarchical Retail Federated Learning (HRFL) with a two-layer neural network.*

<p align="center">
<img width="700" height="300" alt="Image" src="https://github.com/user-attachments/assets/a748dfd1-a4b4-49da-89c8-ff30ade582c1" />
</p>

- **FEDRETAIL** outperforms all baselines, achieving the highest accuracy (~0.90) and lowest loss.
- **Centralized FL** performs well but slightly below FEDRETAIL.
- Individual retailers (Retailer 1, 3, 9) show fluctuating accuracy and higher loss due to limited local data.

---

### 📊 HRFL with Softmax Regression 

> 📌 **HRFL Softmax Plot**  
> **Caption:** *Accuracy and loss comparison during HRFL training using softmax regression.*

<p align="center">
<img width="700" height="300" alt="Image" src="https://github.com/user-attachments/assets/27889f4a-64ee-466d-934f-133014d816f5" />
</p>

- **FEDRETAIL** again leads in accuracy and maintains the lowest loss.
- **Centralized FL** shows competitive performance but doesn’t match FEDRETAIL’s convergence.
- Retailer-specific models converge slower and exhibit higher loss.

---

### 🧪 Participation Probability Impact – Softmax Regression 

> 📌 **Participation Probability Plot (Softmax)**  
> **Caption:** *Impact of varied participation probabilities on federated convergence using softmax regression.*

<p align="center">
<img width="700" height="300" alt="Image" src="https://github.com/user-attachments/assets/86d84114-a60b-4eec-9985-bc410a154cd5" />
</p>

- Lower participation (0.09) leads to slower accuracy gains and higher initial loss.
- Higher participation (0.7) accelerates convergence and stabilizes training.
- Centralized FL maintains superior loss profile throughout.

---

### 🧪 Participation Probability Impact – Neural Network 

> 📌 **Participation Probability Plot (NN)**  
> **Caption:** *Impact of varied participation probabilities on federated convergence using a two-layer neural network.*

<p align="center">
<img width="700" height="300" alt="Image" src="https://github.com/user-attachments/assets/ebb57bb3-70c5-4eac-8115-2ff1142d7102" />
</p>

- Participation probability directly influences convergence speed and final accuracy.
- FEDRETAIL with 0.7 participation shows near-centralized performance.
- FEDRETAIL with 0.09 starts slow and suffers from high loss early on.

---

### 🔑 Key Findings

- ✅ **FEDRETAIL consistently outperforms centralized and retailer-specific models across architectures.**
- 📈 **Higher participation probabilities yield faster convergence and better generalization.**
- 🧠 **FedAvg remains a strong baseline under IID conditions.**
- 🔁 **FSVRG enhances gradient stability, while CO-OP optimizes communication.**
- 🏪 **HRFL enables collaborative uplift without compromising data privacy.**


---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/<your-username>/FEDRETAIL.git
cd FEDRETAIL
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Experiments

### Algorithm Comparison

```bash
python experiments/algorithm_comparison.py
```

### Horizontal Federated Learning

```bash
python experiments/hfl_experiments.py
```

### Vertical Federated Learning

```bash
python experiments/vfl_experiments.py
```

Each script:

* Loads data
* Performs partitioning
* Trains federated models
* Evaluates performance
* Generates plots

---

## 🔐 Security & Privacy

* No raw data exchange between clients
* Local-only training enforced
* Placeholder authentication layer included
* Architecture compatible with:

  * Secure aggregation
  * Differential privacy
  * Homomorphic encryption (future work)

---

## 📄 Publication & IP

* **Journal:** *International Journal of Advanced Research in Science, Communication and Technology (IJARSCT)*
* **DOI:** 10.48175/IJARSCT-16909
* **Software Copyright (Government of India):** SW-18815/2024

<p align="center">
<img width="500" height="800" alt="Image" src="https://github.com/user-attachments/assets/6c8b2f48-0ab6-4e80-9b6a-45ea9be3e51b" />
</p>

---

## 📜 Citation

```bibtex
@article{chongder2024fedretail,
  title={FEDRETAIL: Federated Retail Data Analysis and Learning Framework},
  author={Chongder, Sumit},
  journal={IJARSCT},
  year={2024}
}
```

---

## 👨‍💻 Author

**Sumit Chongder**

🎓 Indian Institute of Technology (IIT) Jodhpur

🔬 Quantum Machine Learning · Quantum Computing · Quantum Key Distribution · Federated Learning · Distributed Systems · Privacy-Preserving ML · Artificial Intelligence · Cloud Computing · Network Security

🔗 Linkedin: [https://www.linkedin.com/in/sumit-chongder/](https://www.linkedin.com/in/sumit-chongder/)

> *Engineering scalable intelligence without compromising data ownership.*

---

## ⭐ Why FEDRETAIL Matters

✔ Research-grade
✔ Industry-ready
✔ Modular & extensible
✔ Privacy-preserving by design
✔ Recruiter-impressive

If this repository helped you, please ⭐ star it — it supports open research.

---

> **FEDRETAIL bridges the gap between data privacy and large-scale retail intelligence — enabling the future of E-commerce 5.0.** 🚀
