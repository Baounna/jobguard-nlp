---
marp: true
theme: default
class: invert
paginate: true
size: 16:9
style: |
  section {
    background: linear-gradient(135deg, #0a0e1a 0%, #131826 100%);
    color: #f0f4ff;
    font-family: 'Inter', sans-serif;
    padding: 60px 80px;
  }
  h1, h2, h3 { color: #ffffff; letter-spacing: -0.01em; }
  h1 { font-size: 2.5rem; }
  h2 { font-size: 1.8rem; border-bottom: 2px solid #7c3aed; padding-bottom: 0.3rem; }
  strong { color: #c4b5fd; }
  table { font-size: 0.85rem; }
  th { background: rgba(124,58,237,0.15); color: #c4b5fd; }
  td, th { border: 1px solid #232a3d; padding: 0.5rem 0.8rem; }
  code { background: #1a2033; color: #c4b5fd; padding: 2px 6px; border-radius: 4px; }
  blockquote { border-left: 4px solid #7c3aed; color: #a8b2cf; padding-left: 1rem; }
  .muted { color: #6b7693; font-size: 0.9rem; }
  .accent { color: #7c3aed; font-weight: 700; }
  .pill { display: inline-block; background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.4); border-radius: 99px; padding: 0.2rem 0.7rem; font-size: 0.8rem; color: #c4b5fd; margin-right: 0.4rem; }
  .legit { color: #10b981; font-weight: 700; }
  .fraud { color: #ef4444; font-weight: 700; }
---

<!-- _class: invert -->

# 🛡️ Job Offer Scam Detector

### A complete classical-NLP pipeline for detecting fraudulent job postings

<br>

**Mohamed Baounna** &nbsp;·&nbsp; **Zakaria Birani**
SIIA S6 &nbsp;·&nbsp; NLP Unit &nbsp;·&nbsp; **Pr. H. El Hamdaoui**
FP Khouribga &nbsp;·&nbsp; May 2026

<br>

<span class="pill">EMSCAD · 17 880 postings</span>
<span class="pill">F1 = 0.892</span>
<span class="pill">SVM + TF-IDF</span>

---

## 🎯 The problem

Online job portals are infested with scams:

- 🚨 **"Earn $5000/week from home"** — vague salary promises
- 🚨 **Reshipping / payment-handler** schemes
- 🚨 **Data-entry typing scams** harvesting personal details
- 🚨 **MLM / Brand Partner** pitches mimicking real jobs

> **Goal:** automatically classify any job posting as legitimate or fraudulent, with **interpretable evidence**.

> **Constraint:** classical ML only. Transformers permitted *for comparison only*.

---

## 📊 Dataset — EMSCAD

| Metric | Value |
|---|---|
| Total postings | **17,880** |
| Legitimate | 17,014 (95.16%) |
| **Fraudulent** | **866 (4.84%)** |
| Imbalance ratio | 1 : 19 |
| Source | Kaggle (`shivamb/real-or-fake-fake-jobposting-prediction`) |
| Fields used | title, company_profile, description, requirements, benefits |

> Imbalance means **accuracy is misleading** — we report **F1** and **ROC-AUC**.

---

## 🧹 Preprocessing pipeline

Each step is **explicitly justified**:

1. **Combine columns** → max info per document
2. **Lowercase** → vocabulary normalisation
3. **Remove URLs / HTML / special chars** → reduce noise
4. **Tokenization** (NLTK) → analysable units
5. **Stop-word removal** → drops uninformative tokens (~30%)
6. **Lemmatization** (WordNet) → `running → run`, smaller vocabulary

> Output: clean, lemmatized token strings, ready for vectorization.

---

## 🔤 Representation comparison (3-way)

| Method | Type | Vocab | F1 (LR) | ROC-AUC |
|---|---|---:|---:|---:|
| **BoW** (CountVectorizer) | sparse counts | 15 000 | 0.825 | 0.980 |
| **TF-IDF** (1–2 grams, sublinear) | sparse weighted | 15 000 | 0.793 | **0.986** |
| **LSA** (TruncatedSVD ×100) | dense semantic | 100 | 0.381 | 0.938 |

**Decision: TF-IDF.** Why?
- Highest **ROC-AUC** → better-calibrated ranking
- IDF amplifies rare scam keywords (`signing bonus`, `data entry`)
- Interpretable feature importance
- LSA **erases** the rare-token signal we need

---

## 🤖 Model comparison (4 variants)

| Model | Accuracy | Precision | Recall | **F1** | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| **SVM + TF-IDF** ⭐ | 0.988 | **0.978** | 0.775 | **0.865** | 0.985 |
| LR + TF-IDF | 0.977 | 0.704 | **0.908** | 0.793 | 0.986 |
| SVM + LSA | 0.966 | 0.821 | 0.370 | 0.510 | 0.940 |
| LR + LSA | 0.867 | 0.246 | 0.844 | 0.381 | 0.938 |

> **SVM** maximises precision. **LR** maximises recall. **LSA** loses signal.

---

## 🎚️ Threshold tuning — the right operating point

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| Default 0.50 | 0.978 | 0.775 | 0.865 |
| **Best 0.30** ✅ | 0.931 | 0.855 | **0.892** |

<br>

> Lifting recall from 77% → 86% costs only 5 points of precision.
>
> For a fraud detector, **a missed scam costs more than a false alarm**.

> Final deployed model: **SVM + TF-IDF @ threshold 0.30, F1 = 0.892**.

---

## 🔍 What did the model learn?

**Top fraud bigrams** (positive SVM coefficients):

<span class="pill">data entry</span>
<span class="pill">work home</span>
<span class="pill">signing bonus</span>
<span class="pill">typing data</span>
<span class="pill">earn</span>
<span class="pill">office manager</span>

**Top legit bigrams** (negative coefficients):

<span class="pill">client</span>
<span class="pill">recruitment</span>
<span class="pill">government</span>
<span class="pill">interview</span>
<span class="pill">team</span>
<span class="pill">english</span>

> The features match **human intuition** — that's the strength of classical ML.

---

## 🖥️ Live demo — JobGuard

The Streamlit app supports **three input modes**:

1. ✏️ **Paste text** — title, company, description, requirements, benefits
2. 📷 **Upload screenshot** — Tesseract OCR fills the description automatically
3. ⚡ **Quick samples** — built-in fraud/legit buttons in the sidebar

**Beyond the assignment:**

- 🛡️ **Scope gate** — refuses non-job-posting inputs
- 🧠 **39-topic detector** — Food, Login Page, Sports, Crypto, Yoga, …
- 📊 **Confidence breakdown** — calibrated probabilities, not just labels

---

## 🧠 Robustness — out-of-domain inputs

What happens if the user pastes a random screenshot (e.g. a login page)?

- The classifier was trained on **job postings only**.
- A naïve binary model would predict <span class="legit">LEGITIMATE 95%</span> by default — wrong.

**Our fix:** a **scope gate** that runs *before* the classifier:

- Counts job-posting keywords in the input.
- If below threshold → bypasses the model.
- Shows a **topic detection card** instead.
- Falls back to *"Out of model's memory"* with a clear explanation when no topic matches.

> **Honest UX > forced prediction.**

---

## 🎬 Live demo flow (5 min)

1. Built-in **Legit Sample** button → ✅ <span class="legit">LEGITIMATE 99%</span>
2. Built-in **Fraud Sample** button → 🚨 <span class="fraud">FRAUDULENT 98%</span>
3. **Real-world legit** posting (Stripe Backend Engineer) → ✅ 99%
4. **Real-world scam** ("data entry, earn $25/hr") → 🚨 99.9%
5. **Subtle MLM** ("Brand Partner") → ⚠️ borderline
6. **Random non-job text** ("CHAWARMA") → 🍔 *Food / Restaurant / Delivery*
7. **Login page screenshot** → 🔐 *Login / Authentication Page*

---

## ⚠️ Limitations & honest discussion

| Limitation | Status |
|---|---|
| ~22% of MLM scams missed (default threshold) | Mitigated by threshold tuning + ack'd in error analysis |
| English-only main vocabulary | Topic detector partially multilingual |
| No contextual understanding (BoW limit) | BERT proposed as future work |
| Vocabulary drift over time | Out of scope for this project |

> **Error-analysis breakdown** of the 39 false negatives:
> 40% MLM, 30% vague international, 15% reshipping, 15% noise.

---

## 🚀 BERT comparison (optional, ready)

Code-ready cell in `4_modeling.ipynb` (§9):

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-MiniLM-L6-v2')
X_bert = encoder.encode(texts)
LogisticRegression().fit(X_bert, y)
```

| Aspect | Classical (deployed) | BERT (proposed) |
|---|---|---|
| F1 | 0.892 | est. +3-6 pts |
| Inference | ~5 ms | ~50 ms |
| Artefact size | <1 MB | ~90 MB |
| Interpretability | ✅ high | ❌ low |
| Assignment-compliant | ✅ yes | ⚠️ comparison only |

> Best engineering choice: **TF-IDF first, BERT for borderline cases**.

---

## 📁 Project structure

```
NLP PROJECT/
├── app.py                    # Streamlit demo (~700 lines, themed dark UI)
├── src/
│   ├── preprocessing.py      # NLTK pipeline
│   ├── representation.py     # TF-IDF / LSA / Word2Vec
│   └── model.py              # SVM, LR, evaluation
├── notebooks/
│   ├── 1_exploration.ipynb   # Dataset analysis
│   ├── 2_preprocessing.ipynb # Each step justified
│   ├── 3_representation.ipynb# BoW vs TF-IDF vs LSA
│   └── 4_modeling.ipynb      # SVM vs LR + threshold + BERT
├── models/                   # Saved artefacts (SVM + vectorizer)
├── data/                     # Raw EMSCAD + 13 generated charts
├── FINAL_REPORT.md           # Methodology, results, discussion
├── SLIDES.md                 # This deck
└── demo_samples.md           # Live-demo cheat sheet
```

---

## ✅ What I learned

- **Preprocessing decisions cascade** — lemmatization vs stemming changes downstream interpretability.
- **Imbalanced detection ≠ "just maximise accuracy"** — F1, AUC, threshold tuning all matter.
- **Classical ML is competitive** — F1 = 0.89, 5 ms latency, < 1 MB model, fully interpretable.
- **Honest UX is engineering** — refusing to predict on out-of-domain inputs is more useful than a confident wrong answer.

---

## 🙏 Thank you

Questions?

<br>

**Mohamed Baounna** &nbsp;·&nbsp; **Zakaria Birani**
SIIA S6 · NLP · Pr. H. El Hamdaoui · FP Khouribga · May 2026

<br>

<span class="pill">F1 = 0.892</span>
<span class="pill">EMSCAD 17 880</span>
<span class="pill">SVM + TF-IDF</span>
<span class="pill">39-topic scope gate</span>
<span class="pill">OCR upload</span>
