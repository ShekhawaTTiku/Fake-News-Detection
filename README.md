<div align="center">

# 📰 Fake News Detection

### Comparing NLP Models on the LIAR2 Fact-Checking Dataset

*Explore how text, fact-checker justifications, and speaker history affect binary claim classification with traditional machine learning, an LSTM, and BERT.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM%20%26%20BERT-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-LIAR2-4C8C4A?style=for-the-badge)](https://github.com/chengxuphd/liar2)

**[Overview](#-overview) · [Dataset](#-dataset-and-labels) · [Results](#-recorded-results) · [Setup](#-installation) · [Notebooks](#-notebook-workflow)**

</div>

---

## ✨ Overview

This project studies binary classification of political claims using the **LIAR2** dataset. It compares a TF–IDF Support Vector Machine (SVM), Random Forest, bidirectional LSTM, and a BERT-based classifier. The traditional and LSTM pipeline combines text features with speaker-credibility ratios; the BERT notebook combines a text representation with credibility features.

The notebooks cover dataset exploration, preprocessing, model training, SHAP explanations for the SVM, and an interactive prediction demonstration.

| 🗞️ Input | 🧠 Models | 📊 Evaluation | 🔎 Interpretation |
|:---:|:---:|:---:|:---:|
| LIAR2 claim, justification, and speaker history | SVM, Random Forest, BiLSTM, and BERT | Weighted F1 and classification reports | SHAP feature analysis for the SVM |

> **Important scope note:** LIAR2 includes justifications written by fact-checkers. Since this project uses the justification as an input feature, the reported scores do not describe a system that can verify a newly published claim before fact-checking. The model learns dataset labels; it does not establish whether a real-world statement is true.

---

## 🚀 Features

- 🧹 Text cleanup, lemmatization, stopword processing, and TF–IDF features.
- 🗣️ Speaker-history credibility ratios joined to text features.
- ⚖️ Comparison of classical, recurrent, and transformer-based models.
- 🧪 A BERT fine-tuning notebook using `bert-base-uncased` and up to 256 tokens.
- 🔍 SHAP analysis of SVM predictions and feature contributions.
- 🖥️ A notebook demonstration for entering a claim and speaker-history values.
- 📊 An optional exploratory data-study notebook.

---

## 📋 Dataset and Labels

The project uses the [LIAR2 dataset](https://github.com/chengxuphd/liar2), derived from PolitiFact fact-checks. The repository includes `train.csv`, `valid.csv`, and `test.csv` in its root.

The original six truthfulness ratings are collapsed into two labels:

| Project label | Original LIAR2 ratings |
|---|---|
| `0` — **FAKE** | `pants-fire`, `false`, `barely-true` |
| `1` — **REAL** | `half-true`, `mostly-true`, `true` |

| Split | Samples | FAKE (`0`) | REAL (`1`) |
|---|---:|---:|---:|
| Train | 18,369 | 10,591 | 7,778 |
| Validation | 2,297 | 1,325 | 972 |
| Test | 2,296 | 1,323 | 973 |

These labels are a simplified grouping of a graded rating scale. In particular, “REAL” does not mean independently verified as wholly true, and “FAKE” combines three different levels of falsity.

---

## 🔄 Modeling Pipeline

```text
LIAR2 train / validation / test CSV files
                    │
                    ▼
       Clean columns and handle missing values
                    │
                    ▼
    Map six ratings to FAKE (0) or REAL (1)
                    │
                    ▼
       Derive speaker-history ratios
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 Cleaned text               Raw text
 TF–IDF (50K terms)          BERT tokenizer
        │                       │
 Join credibility data       CLS representation
        │                       │
 SVM / RF / BiLSTM          BERT + credibility
        └───────────┬───────────┘
                    ▼
        Predicted dataset label and score
```

The TF–IDF pipeline uses up to **50,000 features** with unigrams and bigrams, then appends speaker-credibility values. The BERT notebook tokenizes raw text to a maximum of **256 tokens** and combines the text representation with credibility features.

---

## 📊 Recorded Results

The current project summary reports the following **test weighted F1** scores:

| Model | Feature approach | Reported test weighted F1 |
|---|---|---:|
| SVM | TF–IDF + speaker credibility | **0.8684** |
| Random Forest | TF–IDF + speaker credibility | **0.8588** |
| BiLSTM | Token sequences + speaker credibility | **0.8521** |
| BERT | BERT text representation + speaker credibility | **0.8830** |

These are project-recorded results, not newly reproduced benchmarks. The notebook also prints split-specific classification reports; compare the test reports when evaluating a run. The use of fact-checker justifications is a major limitation when interpreting these figures.

---

## ⚙️ Requirements

Use **Python 3.9–3.11** as noted by the project. The repository does not currently include a pinned `requirements.txt`; install the notebook dependencies in your environment as needed:

```bash
python -m pip install jupyter pandas numpy scipy scikit-learn nltk matplotlib seaborn shap torch transformers
```

The BERT notebook may need a GPU for practical fine-tuning time. It can run on CPU, but training may take several hours. Notebook cells may download NLTK resources such as stopwords and WordNet data when first run.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ShekhawaTTiku/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create an Environment and Install Dependencies

```bash
python -m venv .venv
```

Activate the environment, then run the installation command from [Requirements](#-requirements). In Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Run the Notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Keep the three CSV files in the repository root so the notebook paths resolve.

---

## ▶️ Notebook Workflow

| Order | Notebook | Purpose |
|---:|---|---|
| 1 | `Preprocessing , ML models (SVM and RF) and LSTM.ipynb` | Main pipeline: inspect data, preprocess text, derive speaker features, train SVM, Random Forest, and BiLSTM. |
| 2 | `BERT.ipynb` | Fine-tune the BERT hybrid classifier on raw claim/justification text and speaker features. |
| 3 | `SHAP.ipynb` | Explain the SVM feature contributions on a sample of test data. |
| 4 | `demonstration.ipynb` | Try model predictions on manually supplied inputs. |
| Optional | `Data Study.ipynb` | Independent exploratory analysis and visualizations. |

Training notebooks save artifacts under `models/`. These generated model files are not included in the repository checkout, so run the relevant training notebook before using the demonstration notebook or SHAP analysis. The demonstration expects the model classes to be defined before loading the saved LSTM and BERT weights.

---

## 📁 Project Structure

```text
Fake-News-Detection/
├── README.md
├── train.csv
├── valid.csv
├── test.csv
├── Preprocessing , ML models (SVM and RF) and LSTM.ipynb
├── BERT.ipynb
├── SHAP.ipynb
├── demonstration.ipynb
├── Data Study.ipynb
└── models/                         # Created by training notebooks; not committed
```

---

## ⚠️ Limitations

| Area | Limitation |
|---|---|
| Input timing | Fact-checker justification is used as an input, so the approach is not a pre-publication or breaking-news detector. |
| Label meaning | Two classes merge six graded truthfulness labels and should not be read as ground-truth truth/false judgments. |
| Dataset scope | Results reflect LIAR2 and its source conventions; they may not generalize to other topics, languages, platforms, or time periods. |
| Reproducibility | Dependency versions and random seeds are not consistently captured in a project requirements lockfile. |
| Artifacts | Trained models and SHAP outputs are generated locally and are not provided in this checkout. |
| Interpretation | SHAP explains model behavior for the selected features; it does not establish factual correctness or causation. |

---

## 🛣️ Future Improvements

- 🧪 Add a pinned environment file and record notebook/runtime versions.
- 🔁 Make preprocessing, label mapping, and evaluation reproducible in reusable scripts.
- 🧾 Compare a statement-only setup separately from the justification-assisted setup.
- 📉 Report per-class precision, recall, confusion matrices, and calibration alongside weighted F1.
- 📦 Provide downloadable model artifacts or a lightweight inference entry point.
- 🧭 Evaluate on external datasets and later claims to assess distribution shift.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python / Jupyter** | Notebook-based data exploration and experiments |
| **pandas / NumPy / SciPy** | Tabular processing and sparse feature assembly |
| **scikit-learn** | TF–IDF, SVM, Random Forest, preprocessing, and evaluation |
| **PyTorch** | BiLSTM and BERT model training |
| **Hugging Face Transformers** | BERT tokenizer and pretrained model components |
| **NLTK** | Token/text preprocessing resources |
| **SHAP** | SVM feature explanations |
| **Matplotlib / Seaborn** | Notebook visualizations |

---

## 📚 References

- [LIAR2 dataset repository](https://github.com/chengxuphd/liar2)
- [BERT base uncased model](https://huggingface.co/bert-base-uncased)
- [SHAP](https://github.com/slundberg/shap)
- [Survey of fake-news detection research](https://dl.acm.org/doi/10.1145/3395046)

---

## 👤 Maintainer

**Digvijay Singh Shekhawat**

[![GitHub](https://img.shields.io/badge/GitHub-@ShekhawaTTiku-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ShekhawaTTiku)

---

<div align="center">

*An educational NLP experiment on fact-checked political claims.*

</div>
