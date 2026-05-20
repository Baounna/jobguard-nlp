# 🕵️ Job Offer Scam Detection — NLP Project

**Students:** Mohamed Baounna · Zakaria Birani  
**Major:** SIIA (S6)  
**Unit:** Natural Language Processing  
**Professor:** Pr. H. El Hamdaoui  
**Deadline:** 20 May 2026  

---

## 📌 Project Overview

This project builds a complete NLP pipeline to automatically classify job postings as **legitimate or fraudulent** based on their text content. The system uses classical Machine Learning algorithms (no Transformers) and is deployed as an interactive web application using **Streamlit**.

---

## 🎯 Objective

Detect scam job offers by analyzing linguistic patterns in job posting texts such as:
- Vague or exaggerated salary promises
- Missing or suspicious company information
- Urgent or manipulative language
- Poor grammar and generic descriptions

---

## 📂 Project Structure

```
job-scam-detection/
│
├── data/
│   ├── raw/                  # Original dataset from Kaggle
│   └── processed/            # Cleaned and preprocessed data
│
├── notebooks/
│   ├── 1_exploration.ipynb   # Data exploration & visualization
│   ├── 2_preprocessing.ipynb # Text cleaning & tokenization
│   ├── 3_representation.ipynb# TF-IDF vs Word2Vec comparison
│   └── 4_modeling.ipynb      # SVM vs Logistic Regression
│
├── src/
│   ├── preprocessing.py      # Text cleaning functions
│   ├── representation.py     # TF-IDF and Word2Vec vectorizers
│   └── model.py              # Training and evaluation logic
│
├── models/
│   ├── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer
│   └── svm_model.pkl         # Saved best model
│
├── app.py                    # Streamlit demo interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 📊 Dataset

- **Name:** Employment Scam Aegean Dataset (EMSCAD)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/whenamancodes/real-or-fake-jobs)
- **Size:** ~17,880 job postings
- **Labels:** `fraudulent` (1) or `legitimate` (0)
- **Key columns used:** `title`, `company_profile`, `description`, `requirements`, `benefits`

---

## 🔧 Pipeline Steps

### 1. Data Collection & Exploration
- Download EMSCAD dataset from Kaggle
- Explore class distribution (imbalanced: ~5% fraud)
- Visualize word clouds, text length distributions
- Identify missing values and handle them

### 2. Text Preprocessing ✅ (Justified)
- **Combine** relevant text columns into one
- **Lowercasing** — normalize text
- **Remove punctuation & special characters**
- **Tokenization** — split text into words
- **Stop word removal** — remove common words (the, is, etc.)
- **Lemmatization** — reduce words to base form (running → run)

> **Justification:** These steps reduce noise and help the model focus on meaningful words that distinguish scam from real posts.

### 3. Text Representation 🔄 (Comparison)

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **TF-IDF** | Weights words by frequency and importance | Fast, interpretable | Ignores word order & semantics |
| **Word2Vec** | Dense vector embeddings (semantic meaning) | Captures context | Slower, needs more data |

> **Chosen method:** TF-IDF (better performance on this task with classical ML)

### 4. Model Building 🤖 (Comparison)

| Algorithm | Description |
|-----------|-------------|
| **SVM** | Finds optimal hyperplane to separate classes |
| **Logistic Regression** | Probabilistic classifier, fast and interpretable |

> **Chosen model:** SVM with TF-IDF (best F1-score on test set)

### 5. Evaluation 📈
- **Accuracy** — overall correct predictions
- **F1-Score** — balances precision and recall (important for imbalanced data)
- **Confusion Matrix** — visualize false positives/negatives
- **ROC-AUC Curve** — model performance across thresholds

### 6. Analysis & Discussion
- Compare TF-IDF vs Word2Vec results
- Compare SVM vs Logistic Regression results
- Discuss misclassified examples
- Limitations and future improvements

---

## 🖥️ Demo — Streamlit Interface

The app allows users to:
1. Paste any job posting text
2. Click **"Analyze"**
3. Get instant prediction: ✅ Legitimate or 🚨 Fraudulent

### Run the app:
```bash
streamlit run app.py
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/job-scam-detection.git
cd job-scam-detection

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
nltk
gensim
streamlit
matplotlib
seaborn
wordcloud
joblib
```

---

## 📝 Deliverables

- [x] README.md
- [x] Jupyter Notebooks (exploration → modeling)
- [x] Final Report (`FINAL_REPORT.tex` → compile to PDF, or `FINAL_REPORT.md`)
- [x] Streamlit Demo Interface (`app.py` — with OCR + scope gate + 39-topic detector)
- [x] Oral Presentation slides (`SLIDES.md` — Marp-compatible)
- [x] Demo cheat sheet (`demo_samples.md`)

## 📈 Final Results

| Model | F1-Score | ROC-AUC | Notes |
|---|---:|---:|---|
| **SVM + TF-IDF** ⭐ (threshold 0.30) | **0.892** | 0.985 | Deployed model |
| SVM + TF-IDF (default 0.5) | 0.865 | 0.985 | Default operating point |
| LR + TF-IDF | 0.793 | 0.986 | Higher recall |
| SVM + LSA | 0.510 | 0.940 | LSA loses signal |
| LR + LSA | 0.381 | 0.938 | Worst variant |

---

## 🚀 Future Improvements

- Add Arabic/French job posting support
- Use BERT for comparison (as allowed by assignment)
- Deploy the app online via Streamlit Cloud

---

*Faculté Polydisciplinaire de Khouribga — SIIA S6 — NLP Unit 2026*