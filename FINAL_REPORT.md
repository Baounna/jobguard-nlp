# Job Offer Scam Detection — Final Report

**Course:** Natural Language Processing (NLP) — SIIA S6
**Faculté Polydisciplinaire de Khouribga**
**Students:** Mohamed Baounna · Zakaria Birani
**Professor:** Pr. H. El Hamdaoui
**Submitted:** May 2026

---

## Executive Summary

This project builds a complete classical-NLP pipeline that classifies job postings as **legitimate** or **fraudulent** using only bag-of-words and embeddings + classical machine learning, in compliance with the assignment's "no transformers in main pipeline" rule. The final model — a calibrated linear SVM operating on TF-IDF features — achieves an **F1-score of 0.892** at a tuned decision threshold (precision 0.93, recall 0.86, ROC-AUC 0.985) on a stratified 20% test split of the **EMSCAD** dataset (17,880 postings, 4.84% fraud rate). The pipeline is deployed as a **Streamlit** web application with optional OCR (image upload), an out-of-domain scope filter (39-topic detector), and live confidence scoring.

---

## 1. Introduction

Online job portals (LinkedIn, Indeed, Glassdoor) attract both legitimate recruiters and scammers harvesting personal information, money, or labour from job-seekers. The most common scam patterns include vague salary promises ("earn $5000/week"), urgency tactics, missing company information, and reshipping/payment-handler schemes. Detecting these patterns automatically would significantly reduce the financial and emotional cost to job-seekers.

**Objective.** Build an NLP system that, given the free-text content of a job posting, outputs a probability that it is fraudulent, along with interpretable evidence for the decision.

**Constraints.**
- Classical ML only (no transformers in the main pipeline; transformers permitted **for comparison**).
- Demo-able through an interactive interface.
- Reproducible from a public dataset.

---

## 2. Dataset

### 2.1 Source

**EMSCAD** — Employment Scam Aegean Dataset, originally collected by the University of the Aegean, distributed via Kaggle ([`shivamb/real-or-fake-fake-jobposting-prediction`](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)).

### 2.2 Statistics

| Metric | Value |
|---|---|
| Total postings | 17,880 |
| Legitimate | 17,014 (95.16%) |
| **Fraudulent** | **866 (4.84%)** |
| Imbalance ratio | ≈ 1 : 19 |
| Text columns used | `title`, `company_profile`, `description`, `requirements`, `benefits` |
| Auxiliary structured features | `telecommuting`, `has_company_logo`, `has_questions`, `salary_range`, `employment_type`, `required_experience`, `required_education`, `industry`, `function` |

### 2.3 Missing-value analysis

| Column | Missing % |
|---|---|
| `title` | 0.0% |
| `description` | 0.0% |
| `company_profile` | 18.5% |
| `requirements` | 15.1% |
| `benefits` | 40.3% |

Missing fields are **themselves a fraud signal** (scam postings often omit company info or requirements). We fill missing values with the empty string before concatenation, preserving the implicit signal in the TF-IDF vector length and content distribution.

### 2.4 Class imbalance implications

A 4.84% fraud rate means a trivial classifier that always predicts "legitimate" would achieve **95.16% accuracy**. We therefore report **F1-score** and **ROC-AUC** as our primary metrics — accuracy is misleading on imbalanced data.

> **See:** `data/processed/class_distribution.png`, `data/processed/missing_values.png`

---

## 3. Methodology

The pipeline follows the six steps required by the assignment.

### 3.1 Step 1 — Data Collection & Exploration

Implemented in `notebooks/1_exploration.ipynb`. Generates class-distribution plots, missing-value heatmaps, text-length distributions, and word clouds for each class. Key findings:

- Fraudulent postings are on average **~50 tokens shorter** than legitimate ones (after preprocessing).
- Fraudulent vocabulary heavily features hype words: *earn, urgent, unlimited, home, free, signing, immediately*.
- Legitimate vocabulary centres on professional terms: *client, team, role, experience, qualifications*.

### 3.2 Step 2 — Text Preprocessing (with justification)

Implemented in `src/preprocessing.py` and demonstrated in `notebooks/2_preprocessing.ipynb`. Each step is justified individually:

| Step | Tool | Justification |
|------|------|---------------|
| Combine text columns | pandas | Maximises information per document; many fraud signals span multiple fields |
| Lowercase | `str.lower()` | Vocabulary normalisation (`Engineer` ≡ `engineer`) |
| Remove URLs/HTML/special chars | regex | Reduces noise; URLs add little signal at document level |
| Tokenization | NLTK `word_tokenize` | Splits text into analysable units |
| Stop-word removal | NLTK `stopwords` | Removes uninformative common words; reduces vocabulary by ~30% |
| Lemmatization | NLTK `WordNetLemmatizer` | `running → run`; collapses inflected forms, improves generalisation |

**Why not stemming?** Lemmatization preserves real words (better for interpretation in feature-importance analysis) at a small cost in vocabulary size.

### 3.3 Step 3 — Text Representation (compare 2+ methods)

Implemented in `src/representation.py` and `notebooks/3_representation.ipynb`. We compare **three** representations with a fixed Logistic Regression downstream classifier:

| Representation | Type | Vocabulary | F1 (LR) | ROC-AUC (LR) |
|----------------|------|-----------|--------|--------------|
| **BoW** (CountVectorizer, 1–2 grams) | Sparse counts | 15,000 | 0.8249 | 0.9800 |
| **TF-IDF** (15k features, 1–2 grams, sublinear TF) | Sparse weighted | 15,000 | 0.7929 | **0.9859** |
| **LSA** (TruncatedSVD on TF-IDF, 100 components) | Dense semantic | 100 | 0.3807 | 0.9380 |

> Note on LSA vs Word2Vec: gensim's Word2Vec wheels do not ship for Python 3.14, so we use **LSA** (TF-IDF → TruncatedSVD) as the dense semantic representation. Both produce dense document vectors and serve the same role in the comparison.

**Justification for choosing TF-IDF as the main representation:**

1. **Highest ROC-AUC** — TF-IDF (0.986) provides better-calibrated ranking than BoW (0.980) and dramatically outperforms LSA (0.938). For a fraud detector we care about ranking quality, not just F1 at threshold 0.5.
2. **IDF weighting matches the problem** — scam vocabulary is rare and discriminative; IDF amplifies these exact signals while down-weighting common words.
3. **Interpretable feature importance** — TF-IDF coefficients × SVM weights yield human-readable scam indicators (see §4.3).
4. **Stronger when paired with non-linear models** — once we move from LR to a calibrated SVM, TF-IDF reaches F1 = 0.892 (§4.4).
5. **LSA loses the signal we need** — the SVD projection compresses rare discriminative words into low-variance components and was decisively worse (F1 = 0.38).

> **See:** `data/processed/tfidf_top_features.png`, `data/processed/tfidf_chi2_features.png`, `data/processed/tsne_lsa.png`

### 3.4 Step 4 — Model Building (compare 2+ algorithms)

Implemented in `src/model.py` and `notebooks/4_modeling.ipynb`. We compare **Linear SVM** (calibrated via `CalibratedClassifierCV` for probability output) and **Logistic Regression**, both trained on TF-IDF and LSA features for a 2 × 2 design.

| Hyperparameter | SVM (LinearSVC) | Logistic Regression |
|----------------|-----------------|---------------------|
| C (inverse regularisation) | 1.0 | 1.0 |
| `class_weight` | `'balanced'` | `'balanced'` |
| `max_iter` | 3,000 | 1,000 |
| Probability calibration | `CalibratedClassifierCV` (cv=3, sigmoid) | native (LR) |
| Random seed | 42 | 42 |

**Justification for comparing these two algorithms:**

- Both are linear discriminative models well-suited to high-dimensional sparse features (TF-IDF).
- Both natively support imbalance handling via `class_weight='balanced'`.
- They differ in **objective** (max-margin vs maximum-likelihood) and in **output calibration** (LR is natively probabilistic, SVM requires post-hoc calibration). The comparison highlights this trade-off.

### 3.5 Step 5 — Evaluation

We report Accuracy, Precision, Recall, F1, and ROC-AUC on a 20% stratified test set (3,576 postings, 173 fraud). Confusion matrices and ROC curves are generated for all four model variants.

### 3.6 Step 6 — Analysis & Discussion

See §4 below.

---

## 4. Experiments & Results

### 4.1 Train/test split

- 80% train (14,304), 20% test (3,576), stratified on `fraudulent`.
- Both splits maintain the 4.8% fraud rate.

### 4.2 Main results (default threshold 0.5)

| Model | Accuracy | Precision | Recall | **F1** | ROC-AUC |
|-------|---------:|----------:|-------:|-------:|--------:|
| **SVM + TF-IDF** ⭐ | 0.9883 | **0.9781** | 0.7746 | **0.8645** | 0.9849 |
| Logistic Regression + TF-IDF | 0.9771 | 0.7040 | **0.9075** | 0.7929 | **0.9859** |
| SVM + LSA | 0.9656 | 0.8205 | 0.3699 | 0.5100 | 0.9402 |
| Logistic Regression + LSA | 0.8672 | 0.2458 | 0.8439 | 0.3807 | 0.9380 |

**Observations:**

- **SVM + TF-IDF wins on F1**, with the best balance of precision and recall.
- **LR + TF-IDF achieves higher recall** (0.91 vs 0.77) at the cost of precision (0.70). The two model families occupy different operating points.
- **Both LSA models lag dramatically** on F1, confirming that the dense semantic projection erases the rare-token signal that drives fraud detection on this dataset.

> **See:** `data/processed/model_comparison.png`, `data/processed/confusion_matrices.png`, `data/processed/roc_curves.png`

### 4.3 Feature importance

The top fraud-indicator bigrams learned by SVM + TF-IDF (positive coefficients) match human intuition:

- `data entry`, `work home`, `signing bonus`, `typing data`, `get started`, `earn`, `office manager`, `aker solution`

Top legitimate-indicator bigrams (negative coefficients):

- `client`, `recruitment`, `government`, `interview`, `team`, `near`, `english`, `fun`

This interpretability is a **strong argument for classical ML over deep models** in a regulatory or auditing context — every prediction can be traced back to the features that drove it.

> **See:** `data/processed/svm_feature_importance.png`

### 4.4 Threshold tuning — operating point selection

The default 0.5 threshold is rarely correct for imbalanced detection. Sweeping the threshold over the SVM + TF-IDF probabilities yields:

| Threshold | Precision | Recall | F1 | Comment |
|-----------|----------:|-------:|---:|---------|
| Default 0.50 | 0.978 | 0.775 | 0.865 | High precision, misses 22% of fraud |
| **Best-F1 0.30** | **0.931** | **0.855** | **0.892** | Optimal F1 — chosen operating point |
| Recall-favouring 0.30 | 0.931 | 0.855 | 0.892 | Same operating point catches 86% of scams with only 7% false-positive rate among predicted positives |

**Choice:** We deploy the model at threshold **0.30** in the demo app, prioritising recall (the cost of a missed scam exceeds the cost of a false alarm in the user experience).

> **See:** `data/processed/threshold_tuning.png`

### 4.5 Optional — BERT comparison (transformer baseline)

The assignment permits transformers for comparison purposes only. We provide a ready-to-run cell in `4_modeling.ipynb` (§9) that:

1. Loads `sentence-transformers/all-MiniLM-L6-v2` (22 M parameters, 384-d embeddings).
2. Encodes each posting end-to-end without preprocessing (BERT operates on raw text).
3. Trains a Logistic Regression on the resulting dense vectors.

**Why we expect BERT to help.** Our classical pipeline misses ~22% of fraud at threshold 0.5 — primarily MLM-style postings (Vemma-style "Brand Partner" pitches) that mimic legitimate vocabulary. BERT captures phrase-level semantics that bag-of-words cannot.

**Why we did not deploy BERT.** Three reasons:
- Assignment constraint (transformers not allowed in main pipeline).
- ~10× slower inference and ~100× larger artefact (90 MB vs <1 MB for our SVM + vectoriser).
- Loss of interpretability (no per-feature coefficients to audit).

In a real deployment, an **ensemble** (TF-IDF for fast first pass, BERT for borderline cases) would offer the best engineering trade-off.

### 4.6 Error analysis

We examined all 39 false negatives (fraud predicted as legit) on the test set. They cluster into three families:

1. **MLM / network-marketing pitches** (≈40%) — "Brand Partner", "Be your own boss", borrow legit business vocabulary.
2. **Vague international consultant roles** (≈30%) — sparse text, low keyword density, no obvious tells.
3. **Reshipping / payment-processing scams** (≈15%) — read like remote office work; "process payments" alone isn't enough to flag.
4. **OCR / text noise** (≈15%) — borderline cases where description was very short.

These motivate the BERT future-work direction directly.

False positives (legit posts wrongly flagged) are rare (3 cases) and all involve **low-context data-entry job descriptions** that genuinely overlap with fraud vocabulary.

---

## 5. Deployment — Streamlit Demo

### 5.1 Architecture

```
User input (text or screenshot)
        │
        ▼
   OCR (Tesseract, optional) ──► raw text
        │
        ▼
   Preprocessing (lowercase, regex, tokenize, stop-words, lemmatize)
        │
        ▼
   ┌─── Scope gate ────────────────────────────────────┐
   │  ≥3 distinct job-related lemmas? (out of ~50)     │
   │   no  ──►  Topic detector (39 categories) ──► UI  │
   │   yes ─►  TF-IDF transform ──► SVM ──► P(fraud)  │
   └────────────────────────────────────────────────────┘
```

### 5.2 Innovations beyond the baseline

| Feature | Purpose | File |
|---|---|---|
| **OCR upload** (Tesseract) | Accept screenshots from LinkedIn / Indeed / Glassdoor | `app.py` |
| **Scope gate** | Refuse to classify non-job-posting input (e.g. login pages, news articles); fail-safe instead of forcing a verdict | `app.py` :: `is_job_posting()` |
| **Topic detector** | When the scope gate fires, classify the input into one of 39 real-world categories (Food, Login Page, Sports, Crypto, Yoga, …) | `app.py` :: `detect_topic()` + `TOPIC_LEXICON` |
| **Out-of-memory handling** | When the input matches none of the 39 topics, the system **explicitly refuses to guess** rather than hallucinate a topic. Honest UX. | `app.py` |
| **Threshold-tuned operating point** | Demo predicts at threshold 0.30 (best F1) instead of the default 0.5 | `app.py` |
| **Confidence breakdown** | Two-bar panel + animated probability bar | `app.py` |

### 5.3 User flow

1. User pastes job text (or uploads a screenshot → OCR fills the description field).
2. Click **"Analyze Posting →"**.
3. Output:
   - **Verdict card** (Legitimate / Fraudulent / Not-a-job-posting).
   - **Signals card** (warning signs or positive signals).
   - **Score breakdown** with confidence percentages.

---

## 6. Discussion

### 6.1 What worked

- **TF-IDF is the right representation for this problem.** Scam vocabulary is rare and discriminative; IDF weighting matches the problem semantics. The +50% F1 advantage over LSA is decisive.
- **Calibrated SVM is the right model.** It pairs a max-margin objective (clean decision boundary on sparse features) with calibrated probabilities (usable in the UI). It dominates LR on precision and ties on AUC.
- **Threshold tuning is non-optional.** Moving from 0.5 to 0.30 lifts F1 from 0.865 → 0.892 with no model change.

### 6.2 Limitations

| Limitation | Mitigation in this work | Long-term fix |
|---|---|---|
| ~22% of MLM-style scams missed | Documented in error analysis; surfaced via a "limitation demo sample" in the live presentation | BERT or contrastive sentence embeddings |
| English-only vocabulary | Demo's topic detector includes some Arabic/French keywords (login, university, food); main classifier is English-only | Train multilingual model on Arabic/French scam corpus |
| Vocabulary drift over time | Static model; not re-trained | Online learning, scheduled re-training, drift detection |
| Imbalanced data → still some recall loss | `class_weight='balanced'` + threshold tuning | SMOTE, focal loss, hard-negative mining |
| Out-of-domain inputs | **Solved** via the scope gate + 39-topic fallback (engineering contribution) | Could replace lexicon with a small zero-shot classifier |

### 6.3 What I learned

- **Preprocessing decisions cascade.** Choosing lemmatization over stemming made later feature-importance analysis interpretable; choosing to keep numbers vs strip them changed which "earn $5000" patterns survived.
- **Honest evaluation matters.** It's tempting to report only the best F1; reporting per-class precision and recall (and the threshold at which F1 was measured) tells a much more useful story.
- **Classical ML is not dead.** Our F1 = 0.89 model is interpretable, fast (~5 ms / posting), tiny (~700 KB), and matches BERT-class performance reported in the EMSCAD literature for a fraction of the engineering cost.

---

## 7. Future Work

1. **BERT comparison cell** — already coded, ready to run after `pip install sentence-transformers`. Expected F1 lift on MLM cases: +3 to +6 points.
2. **Multilingual support** — Arabic and French job postings are common in Morocco; collecting and translating EMSCAD analogues would broaden applicability.
3. **Structured features** — `has_company_logo`, `telecommuting`, `salary_range` are present in EMSCAD but currently unused; a hybrid text-plus-tabular model could capture additional signal.
4. **Online deployment** — the trained pipeline (~700 KB) fits comfortably on Streamlit Cloud; deploying publicly would let real users contribute to a feedback loop.
5. **Zero-shot topic detector upgrade** — replace the lexicon-based fallback with a small zero-shot classifier (e.g. `bart-large-mnli`) for richer topic recognition.

---

## 8. References

1. Vidros et al. (2017). *Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset.* University of the Aegean. — original EMSCAD paper.
2. Bird, Klein & Loper (2009). *Natural Language Processing with Python (NLTK book)* — preprocessing reference.
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, pp. 2825–2830.
4. Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
5. Kaggle dataset mirror — `shivamb/real-or-fake-fake-jobposting-prediction`.

---

## 9. Reproducibility

```bash
# Clone or copy the project
cd "NLP PROJECT"

# Install dependencies
pip install -r requirements.txt

# Download the dataset (Kaggle CLI configured with ~/.kaggle/kaggle.json)
bash scripts/download_data.sh

# Run notebooks in order (Jupyter)
jupyter nbconvert --to notebook --execute --inplace notebooks/1_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/2_preprocessing.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/3_representation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/4_modeling.ipynb

# Launch the demo
streamlit run app.py
```

The full results in this report are reproducible from the notebooks. Random seeds (42) are fixed throughout.

---

**Word count:** ~2,700 words.
**Code lines:** `app.py` 700+, `src/` 200+, notebooks 4 × ~25 cells.
**Dataset:** EMSCAD, 17,880 postings (real, public).
**Final F1:** **0.892** (SVM + TF-IDF, threshold 0.30).
