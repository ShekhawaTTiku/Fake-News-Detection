# NLP-Based Automated Fake News Detection

A personal project building an end-to-end fake news detection system using Natural Language Processing (NLP) and Machine Learning on the LIAR2 dataset. It classifies real-world political statements as Fake or Real by combining text analysis with speaker credibility features.

## 📌 Project Overview
Most fake news datasets label full news articles as simply real or fake. This project uses **LIAR2**, a more realistic and challenging dataset containing:
- Short real-world political claims sourced from PolitiFact.
- A detailed justification written by human fact-checkers.
- Speaker credibility history showing how often the speaker has been rated true, false, pants-on-fire, etc., in the past.

This combination allows the models to learn actual linguistic and contextual cues rather than relying on simple surface-level patterns.

## 📊 Dataset (LIAR2)
**Source:** [LIAR2 Repository](https://github.com/chengxuphd/liar2/tree/main/liar2)  
**Format:** CSV files with 16 columns per row.  

**Label mapping (Binary Classification):**
- **`0 = FAKE`** (Original labels: *pants-fire, false, barely-true*)
- **`1 = REAL`** (Original labels: *half-true, mostly-true, true*)

**Class Distribution:**
| Split | Total Samples | Fake (0) | Real (1) |
| :--- | :--- | :--- | :--- |
| **Train** | 18,369 | 10,591 (58%) | 7,778 (42%) |
| **Validation** | 2,297 | 1,325 (58%) | 972 (42%) |
| **Test** | 2,296 | 1,323 (58%) | 973 (42%) |

## 🧠 Models & Performance
Four models are implemented and compared. All models use **Weighted F1** as the primary metric due to the mild 58/42 class imbalance.

| Model | Type | Features | Test F1 Score |
| :--- | :--- | :--- | :--- |
| **SVM** | Traditional ML | TF-IDF + Credibility | 0.8684 |
| **Random Forest** | Ensemble ML | TF-IDF + Credibility | 0.8588 |
| **BiLSTM** | Deep Learning | Sequences + Credibility | 0.8521 |
| **BERT** | Transformer | Raw Text + Credibility | **0.8830** |

*Note: BERT achieves the best performance. SVM is a surprisingly strong baseline. The strong performance across all models (83-88% F1) is largely due to including the justification column as a feature. Most published results on LIAR use only the statement and achieve 60-70% accuracy.*

## ⚙️ Installation
**Requirements:** Python 3.9, 3.10, or 3.11 is required. *(Python 3.12 and above are not supported).*

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShekhawaTTiku/Fake-News-Detection.git
   cd Fake-News-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: NLTK data is downloaded automatically when notebooks run. To download manually:*
   `python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"`

3. **Add the dataset files**
   Place `train.csv`, `valid.csv`, and `test.csv` in the root project folder. Download them from the [LIAR2 repo](https://github.com/chengxuphd/liar2/tree/main/liar2).

## 🚀 Getting Started (Execution Order)

Run the notebooks in this exact order:

| Order | Notebook | Approx. Time |
| :--- | :--- | :--- |
| 1 | `Preprocessing , ML models (SVM and RF) and LSTM.ipynb` | ~15 min |
| 2 | `BERT.ipynb` | ~20-30 min (GPU) / 3-5 hrs (CPU) |
| 3 | `SHAP.ipynb` | ~60-75 min |
| 4 | `demonstration.ipynb` | Instant |

## 📓 Notebook Details
*   **Preprocessing, ML models (SVM and RF) and LSTM.ipynb:** The main notebook. Covers dataset loading, cleaning, computing speaker credibility ratios, text cleaning (lemmatization, stopword removal), TF-IDF vectorization (50,000 features), and training the SVM, RF, and BiLSTM models. Saves models to the `models/` folder.
*   **BERT.ipynb:** Standalone fine-tuning notebook. Uses raw uncleaned text fed directly into the BERT WordPiece tokenizer (max 256 tokens). Fine-tunes `bert-base-uncased` with a custom hybrid classifier head that concatenates the CLS token output with speaker credibility ratios.
*   **SHAP.ipynb:** Loads the trained SVM model and computes SHAP values for 200 test samples. Produces global feature importance charts showing which words drive predictions and how credibility features contribute.
*   **demonstration.ipynb:** Interactive demo. Loads all models and runs predictions on custom inputs. Allows you to set `STATEMENT`, `JUSTIFICATION`, and speaker history counts to output predictions with confidence scores and a majority vote verdict.
*   **Data Study.ipynb:** Exploratory data analysis and visualizations. Optional and independent.

## 🗂️ Project Files & Models Directory
*   `train.csv`, `valid.csv`, `test.csv` — Datasets
*   `models/svm_model.pkl` — Trained SVM model
*   `models/tfidf_vectorizer.pkl` — Fitted TF-IDF vectorizer
*   `models/robust_scaler.pkl` & `robust_scaler_bert.pkl` — Fitted RobustScalers
*   `models/rf_model.pkl`, `lstm_model.pt`, `bert_classifier.pt` — *(Not Pushed — Retrain Required)*
*   `models/shap_values.pkl` — *(Not Pushed — Recompute Required)*

## 🔄 Pipeline Summary

```text
  Raw CSV data (train / valid / test)
            |
            v
  Column cleanup + null handling
            |
            v
  Binary labels (0=Fake, 1=Real)
            |
            v
  Speaker credibility ratios + RobustScaler
            |
     -------+-------
     |               |
     v               v
  Text cleaning    Raw text
  (SVM/RF/LSTM)    (BERT)
     |               |
     v               v
  TF-IDF          BERT tokenizer
  50K features    max 256 tokens
     |               |
     v               v
  hstack with      CLS output +
  credibility      credibility
  = 50,006 feat.   = 774 feat.
     |               |
     +-------+-------+
             |
             v
     SVM / RF / LSTM / BERT
             |
             v
     Fake or Real + confidence
             |
             v
     SHAP explanation (SVM)
```

## ⚠️ Known Issues
*   **Python version:** Python 3.14+ is not compatible. Use 3.9, 3.10, or 3.11. Check this first if you see `scipy` or `torch` import errors.
*   **SHAP computation time:** Takes 60-75 minutes on CPU for 200 samples. Do not interrupt; values save automatically.
*   **BERT memory:** If you encounter an out-of-memory error on GPU, reduce `BATCH_SIZE` from 64 to 32 in `BERT.ipynb` Block 7.
*   **PyTorch FutureWarning:** A warning about `torch.cuda.amp.autocast` may appear during BERT training. It is cosmetic.
*   **Model loading in demo:** The `LSTMClassifier` and `BertHybridClassifier` classes must be defined (Block 2) before loading model weights in `demonstration.ipynb`.

## 📚 References
*   [LIAR2 Dataset](https://github.com/chengxuphd/liar2)
*   [BERT (Hugging Face)](https://huggingface.co/bert-base-uncased)
*   [SHAP](https://github.com/slundberg/shap)
*   [Fake News Survey](https://dl.acm.org/doi/10.1145/3395046)
