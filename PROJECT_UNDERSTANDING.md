# 🧠 Project Understanding — Defense Preparation Guide

> A metacognitive guide: not "what we built" but "how to *think* about every piece."
> Read this twice. The prof will ask **why**, not **what**.

---

## Mental Model #1 — The Big Picture in 90 seconds

Before any detail, internalize this one paragraph. If the prof opens with *"Explain your project in 1 minute"*, this is what you say:

> *"We built a text classifier that predicts whether a job posting is real or fraudulent. We used a public dataset of 17,880 postings called EMSCAD, where about 5% are scams. Our pipeline turns each posting into a numeric vector with TF-IDF, then a calibrated linear SVM decides if it's fraud. We chose this combination because it gives the highest F1-score among the variants we tested (0.892), and crucially because every prediction is interpretable — we can show exactly which words drove each decision. The whole thing is deployed as a Streamlit web app called JobGuard that also accepts screenshots through OCR."*

**Everything else in this document is just unpacking that paragraph.**

---

## Mental Model #2 — The Problem Has a Specific Shape

When you tell the prof *"this is a fraud detection task"*, she should immediately ask **what makes this specific?**

Here's the shape:

### It's a binary classification task...
- Output: 0 = legitimate, 1 = fraudulent
- One label per posting

### ...with severe class imbalance...
- 95% legit, 5% fraud
- **This is the most important property** of the dataset
- It changes everything about evaluation

### ...where the signal is in vocabulary
- Scammers use distinctive language (`earn`, `urgent`, `unlimited`, `home`, `signing bonus`)
- Legitimate companies use boring corporate vocabulary (`team`, `client`, `experience`, `requirements`)
- This means **word-frequency methods will work well** — no need for deep semantic understanding

### ...with high cost asymmetry
- Missing a scam = user gets defrauded, loses money/identity
- False alarm = user just double-checks a real job
- **Missing a scam is worse than a false alarm**
- This drives our threshold choice later

### ...with explainability requirements
- A scam detector that says "trust me, this is fraud" without evidence is useless
- We need to be able to **show the user why**
- This argues for linear, interpretable models

> **Why this matters in the defense:** if you can articulate the shape of the problem, every later decision becomes obvious. The prof will see you understand the domain, not just the code.

---

## Mental Model #3 — The Dataset Drives Everything

### What is EMSCAD?

**E**mployment **S**cam **A**egean **D**ataset, collected by researchers at the University of the Aegean, distributed publicly via Kaggle. 17,880 real job postings scraped from the web, each manually labeled as legitimate or fraudulent.

Five free-text fields per posting:
- `title` (e.g., "Software Engineer")
- `company_profile` (about the hiring company)
- `description` (the role description)
- `requirements` (skills, qualifications)
- `benefits` (compensation, perks)

### How to think about the missing-values pattern

| Field | Missing % | What this tells us |
|---|---|---|
| title | 0% | Always provided |
| description | 0% | Always provided |
| company_profile | 18.5% | Legit companies usually fill this; scams often skip it |
| requirements | 15.1% | Scams skip requirements ("no experience needed") |
| benefits | 40.3% | Often skipped by both, less signal |

**The missing values are themselves a fraud signal.** When we concatenate the 5 fields into one document, a missing field shrinks the document's length. The TF-IDF vector for a "thin" document looks different from a "rich" document. So we get the missing-field signal for free.

### Why a 5% fraud rate breaks accuracy

If we always predict "legitimate", we get 95% accuracy. The model has done nothing useful — but the metric looks great. This is **why we report F1 and ROC-AUC, not accuracy**.

> **Defense angle:** if the prof asks *"why F1?"*, the answer is the imbalance. F1 balances precision and recall, and a "always predict majority" model gets F1 ≈ 0, exposing it as useless.

---

## Mental Model #4 — Preprocessing Is a Series of Bets

Every preprocessing step is a **deliberate choice with trade-offs**. The prof will ask why for at least one of them.

### Step 1 — Combine the five text columns

**What we do:** Concatenate title + company_profile + description + requirements + benefits into one big string per posting.

**Why:** A scam signal might live in any field. If we trained five separate models, we'd waste data. One unified model can pick up signal from wherever it lives.

**Alternative considered:** Train field-specific models and ensemble them. **Why we didn't:** complexity not justified by the small dataset; linear models already capture cross-field interactions.

### Step 2 — Lowercase

**What:** `Engineer` and `engineer` become the same token.

**Why:** Reduces vocabulary size by ~30%. Improves generalization.

**Alternative:** Case-sensitive. **Why we didn't:** Capitalization rarely carries semantic information in job postings (random capitalization in fraud is noise, not signal).

### Step 3 — Remove URLs, HTML tags, special characters with regex

**Why:** Job postings scraped from websites often contain HTML residue (`&amp;`, `<br>`) or URLs that don't generalize.

**Alternative considered:** Keep URLs as a feature (e.g., `has_suspicious_url`). **Why we didn't:** would need a curated blacklist; out of scope for this project.

### Step 4 — Tokenize with NLTK's `word_tokenize`

**Why:** Splits text into discrete words while handling punctuation correctly (e.g., "U.S." stays one token).

**Alternative:** simple `.split()`. **Why we didn't:** would break on contractions and punctuation-attached words.

### Step 5 — Remove English stop words

**What:** Drop `the`, `is`, `a`, `and`, `or`, etc. — about 30% of vocabulary.

**Why:** Stop words are not discriminative. They appear equally in legit and fraud postings. Removing them shrinks the feature space and lets the model focus on words that actually differ between classes.

**Alternative:** Keep stop words. **Why we didn't:** they add noise without signal; TF-IDF would down-weight them anyway, but explicit removal makes the vocabulary cleaner.

### Step 6 — Lemmatize with WordNet

**What:** `running → run`, `applications → application`, `was → be`.

**Why:** Collapses inflected forms so the model treats them as the same concept.

**Alternative considered:** Stemming (Porter or Snowball). **Why we chose lemmatization instead:** stemming produces non-words (`running → run` is OK, but `engineering → engineer` becomes `engin`). When we later look at feature importance, we want to see real English words so the prof — and a future auditor — can read them. **Interpretability cost vs vocabulary size cost.** We chose interpretability.

> **This is the kind of meta-reasoning the prof wants to see.** Don't say "I lemmatized". Say "I chose lemmatization over stemming because…"

---

## Mental Model #5 — Representation Is a Geometry Choice

This is the hardest concept to articulate. Practice this section out loud.

### What does "text representation" even mean?

Machine learning models work on **vectors of numbers**, not text. We need to turn each posting (a string of words) into a fixed-size numeric vector. The question is: **what should each dimension represent?**

There are three families to choose from:

### Family 1 — Bag-of-Words (BoW)

**Each dimension = one word in the vocabulary. The value = how many times that word appears in this document.**

Example: vocabulary = {`team`, `earn`, `urgent`}. The document "earn urgent earn" becomes the vector `[0, 2, 1]` (zero `team`, two `earn`, one `urgent`).

**Pros:**
- Simple, fast
- Captures vocabulary differences naturally
- Easy to interpret (each dimension is a word)

**Cons:**
- Ignores word order ("not fraudulent" looks identical to "fraudulent not")
- Treats all words equally — `the` (frequent, unhelpful) and `signing bonus` (rare, very helpful) get the same weight
- High dimensional (one dimension per word — 15,000+ dimensions)

### Family 2 — TF-IDF (Term Frequency × Inverse Document Frequency)

**Same dimensions as BoW, but the values are weighted: TF × IDF.**

- **TF (Term Frequency)** = how often this word appears in *this document*
- **IDF (Inverse Document Frequency)** = how rare this word is across *all documents*

Words that are frequent in this document **AND rare in the corpus** get a high score. Words that appear everywhere (like `the`) get a near-zero score.

**Why this matters for fraud detection:** scam vocabulary (`signing bonus`, `data entry`, `work home`) is rare overall but frequent in scam postings. IDF amplifies exactly the signal we need. **This is why TF-IDF beat BoW on ROC-AUC** in our experiments.

**Mathematically:**
```
TF-IDF(word, doc) = TF(word, doc) × log(N / DF(word))
```
where N is the total number of documents and DF is the number of documents containing the word.

### Family 3 — Dense Embeddings (Word2Vec, GloVe, LSA, BERT)

**Each dimension is no longer a word. Documents become dense vectors (e.g., 100 or 300 floats) that capture semantic similarity.**

Words like `salary` and `wage` become similar vectors, even though BoW/TF-IDF would treat them as completely different.

**In our project, we used LSA** (Latent Semantic Analysis) — TruncatedSVD applied to the TF-IDF matrix — as our dense semantic representation. Why LSA and not Word2Vec? **gensim's Word2Vec doesn't ship wheels for Python 3.14 yet**, so we substituted LSA, which serves the same comparative role.

### Why LSA failed in our case

The SVD projection finds the **main axes of variation** in the data. It captures *broad* topical similarity. But fraud detection depends on **rare, specific words** like `signing bonus` and `typing data`. SVD compresses those into noise components because they don't drive most of the variance.

**Visual mental model:** imagine projecting a 3D scatter onto its 2D principal-component plane. If the discriminative information is in the 3rd component (the "small" axis), the projection throws it away. That's exactly what happens to scam vocabulary.

**Numerically:** LSA dropped from F1 = 0.79 (TF-IDF with LR) to F1 = 0.38 (LSA with LR). **More than half the performance disappeared** because we lost the rare-word signal.

### So why is TF-IDF the right choice?

Five-point justification you should be able to recite:

1. **Best ROC-AUC** (0.986) → best probability calibration
2. **IDF amplifies rare scam words** → matches the problem structure
3. **Interpretable** → we can show feature importance to the prof and to the user
4. **Strong with non-linear models** → when paired with calibrated SVM, F1 reaches 0.892
5. **LSA destroyed the signal** → empirical proof that dense embeddings aren't always better

> **Defense angle:** if the prof says *"why didn't you just use embeddings?"*, the answer is the LSA experiment. We tried it and it lost 40 points of F1.

---

## Mental Model #6 — The Two Algorithms Are Geometrically Different

### Logistic Regression (LR)

**What it does:** Models the probability of fraud as a sigmoid of a linear combination of features.

```
P(fraud | x) = sigmoid(w·x + b)
```

It fits `w` (the feature weights) by maximizing the likelihood of the training labels.

**Why it makes sense for this task:** Linear in TF-IDF features. The weights are interpretable: positive weight on `data entry` means the word increases fraud probability.

**Behavior on our data:**
- High **recall** (0.91) — catches lots of scams
- Lower **precision** (0.70) — flags too many legit postings as fraud
- It's eager. It would rather be wrong on a legit posting than miss a scam.

### Support Vector Machine (SVM)

**What it does:** Finds the **hyperplane** that maximally separates the two classes in feature space.

Instead of "what's the probability", it asks "which side of the line is this point on, and how far?"

**Why it differs from LR:**
- LR uses *all* training points to estimate the boundary
- SVM uses only the *support vectors* — the boundary cases near the margin
- This makes SVM more **robust to outliers** and often **more precise**

**Behavior on our data:**
- High **precision** (0.978) — when it says fraud, it's almost always right
- Lower **recall** (0.78) — it misses some borderline scams
- It's cautious. It would rather miss a scam than falsely accuse a legit posting.

### The calibration wrinkle

SVM doesn't natively output probabilities (only "which side of the line"). We wrap it in `CalibratedClassifierCV`, which fits a sigmoid on top of the SVM scores to convert them to calibrated probabilities. Now SVM outputs `P(fraud) ∈ [0, 1]` just like LR. **This is essential for the Streamlit confidence display** to show "99.3% fraud" — without calibration we'd only have raw distances.

### Why we deploy SVM

For a user-facing scam detector, **precision matters more than recall**:
- A false positive damages trust ("JobGuard flagged my real job posting? Useless tool.")
- A false negative is unfortunate but the user can still spot obvious scams themselves
- We can recover some lost recall with threshold tuning (next mental model)

> **Defense angle:** *"why SVM over LR?"* Answer: "precision 97.8% vs 70.4%. In a real fraud detector, falsely accusing a legitimate company costs us more user trust than missing one borderline scam."

---

## Mental Model #7 — Evaluation Is Not One Number

### Why we report 5 metrics

| Metric | What it tells us | Why we need it |
|---|---|---|
| **Accuracy** | % of predictions correct overall | Misleading on imbalanced data |
| **Precision** | When we predict fraud, how often are we right? | Measures false-alarm rate |
| **Recall** | Of all real frauds, how many did we catch? | Measures miss rate |
| **F1** | Harmonic mean of precision and recall | Single-number summary |
| **ROC-AUC** | Ranking quality across all thresholds | Threshold-independent |

You'll trip yourself up if you only memorize the F1 number. **Internalize what each metric is asking.**

### The cross-validation result

A single 80/20 split could be lucky. **5-fold stratified cross-validation** retrains the pipeline 5 times on different folds and reports mean ± std. Our F1 standard deviations are all **under 0.04**, meaning:

> *The deployed F1 of 0.892 is a property of the model, not a property of the lucky test split.*

This is a key defense line. The prof might ask *"how do you know your number is reliable?"* — answer: "the CV standard deviation is below 0.04, so the model is stable across different train/test splits."

---

## Mental Model #8 — Threshold 0.30 Is a Business Decision

### Where the 0.5 default comes from

By convention, classifiers predict "positive" if `P > 0.5`. This is **purely mathematical**. It treats false positives and false negatives as equally bad. In real applications, **they almost never are**.

### How we picked 0.30

We swept the threshold from 0 to 1 in small steps. At each value we recorded precision, recall, and F1. We picked the threshold that maximized F1 — that turned out to be **0.30**.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 (default) | 0.978 | 0.775 | 0.865 |
| **0.30 (chosen)** | **0.931** | **0.855** | **0.892** |

**We traded 5 points of precision to rescue 8 points of recall.** In fraud detection, that's a win.

### Why this lifts the MLM case in the demo

The borderline MLM sample scores P(fraud) ≈ 0.44:
- At threshold 0.50, P = 0.44 → "legit" (slips through)
- At threshold 0.30, P = 0.44 → "fraud" (caught)

The model didn't change. The probability didn't change. **Only the operating point changed**, and it caught the scam.

> **Defense angle:** if the prof asks *"why this specific 0.30?"* — answer: "F1-optimal point on the precision-recall curve. The reason it lifts our F1 is that we're catching 8 more percentage points of scams while only adding 5 percentage points of false alarms. In the cost calculus of fraud detection, that's an obvious win."

---

## Mental Model #9 — The Deployed System Is More Than the Model

Three engineering features in JobGuard that the prof may ask about:

### Feature 1 — The OCR pipeline (Tesseract)

**What:** User can drag a screenshot. Tesseract OCR extracts the text, fills the description field automatically.

**Why it matters:** Most scam screenshots on social media are JPEG images, not selectable text. Without OCR, the user would have to retype them. We **cache the OCR result per file_id** so it runs only once per upload, and we never overwrite manual edits.

### Feature 2 — The scope gate

**What:** Before classifying, we count how many job-related lemmas appear in the input (out of about 50 known terms). If fewer than 3, we refuse to classify.

**Why it matters:** The model was trained only on job postings. If a user pastes a login page, a recipe, or random text, **a naive classifier would still emit a verdict** — usually "legitimate", because legit dominates training. That's dishonest UX.

The scope gate gives us **fail-safe behavior**: we know what we know, and we admit what we don't.

### Feature 3 — The 39-topic detector

**What:** When the scope gate rejects an input, we run a lightweight keyword classifier over 39 real-world topics (Food, Sports, Login Page, etc.). If something matches, we tell the user what their text is actually about.

**Why this is unusual:** Most binary classifiers either predict or refuse. We do better: we predict, refuse with a reason, AND explain what the input *is*. This is the kind of polish that distinguishes a project from a demo.

> **Defense angle:** if the prof asks *"why these engineering features?"* — answer: "to make the system honest. The classifier was trained on a specific distribution. Outside that distribution, instead of hallucinating a confident wrong answer, we either explain what the text is or refuse to guess. That's more useful to a real user than 95% accuracy on the right kind of input plus garbage on everything else."

---

## Mental Model #10 — Limitations Are Strengths in Disguise

The prof might ask *"what doesn't your system do?"* This question is a gift. **Naming a limitation honestly is a sign of expertise.** Three you should be ready with:

### Limitation 1 — MLM scams slip through

~22% of fraud cases at threshold 0.5 (lower at 0.30). The hardest cases are MLM "Brand Partner" pitches that **mimic legitimate business vocabulary**.

**Why it's a limitation, not a bug:** Bag-of-words representations can't distinguish "join our growing team" (legit) from "be your own boss on our team" (MLM). The semantic context that disambiguates them lives at the phrase or sentence level, which **only contextual embeddings (BERT) can capture properly**.

**The right framing:** "Our system is a strong baseline. To close this gap, we'd need contextual embeddings — which the assignment explicitly forbids in the main pipeline, but we have a comparison cell ready."

### Limitation 2 — English-only vocabulary

Our NLTK preprocessing and TF-IDF vocabulary are English. Arabic and French postings (common in Morocco) would not be classified correctly.

**The right framing:** "This is a domain extension, not a fundamental flaw. The architecture is modular. To support Arabic/French, we'd preprocess with multilingual lemmatizers (Stanza or spaCy) and re-train on a multilingual EMSCAD-equivalent."

### Limitation 3 — Vocabulary drift

Scam authors adapt. The vocabulary that flagged scams in 2017 (when EMSCAD was collected) is partially obsolete in 2026.

**The right framing:** "A real deployment would need scheduled retraining and possibly online learning. We documented this as future work."

---

## Mental Model #11 — Why BERT Is Allowed-But-Not-Used

The assignment says: *"Transformers (BERT, GPT, etc.) are not allowed. They may only be used for comparison."*

The trap question: *"Did you actually run a BERT comparison?"*

**Honest answer:** We provide a ready-to-run cell in notebook 4 §9 that loads `sentence-transformers/all-MiniLM-L6-v2`, encodes the postings, and trains an LR on top. **The cell is commented out** because (a) we couldn't `pip install sentence-transformers` in our default environment, and (b) the deployed pipeline must be classical per the assignment.

**The right framing:**

> *"Our main pipeline is fully classical, as required. We provide the BERT comparison code as future work — anyone running `pip install sentence-transformers` can reproduce the comparison in one command. Based on the EMSCAD literature, BERT would lift F1 by 3-6 points on the MLM cases — exactly the family our current model misses."*

This is honest and shows scholarly awareness. **Don't lie and claim you ran it.**

---

## Mental Model #12 — The Numbers You Must Know Cold

Memorize these. If you stumble on a number during the defense, it looks bad.

| Metric / Fact | Value |
|---|---|
| Dataset name | EMSCAD |
| Total postings | 17,880 |
| Legitimate postings | 17,014 |
| Fraudulent postings | 866 |
| Fraud rate | 4.84% |
| Imbalance ratio | 1:19 |
| Train/test split | 80% / 20%, stratified, random_state=42 |
| Train size | 14,304 |
| Test size | 3,576 |
| TF-IDF vocabulary size | 15,000 (1-2 grams) |
| Deployed model | Calibrated Linear SVM + TF-IDF |
| Deployed threshold | 0.30 |
| **Deployed F1** | **0.892** |
| **Deployed ROC-AUC** | **0.985** |
| **Deployed Precision** | **0.931** |
| **Deployed Recall** | **0.855** |
| Inference latency | ~5 milliseconds per posting |
| Model artifact size | < 1 megabyte |
| Number of topics in scope-gate fallback | 39 |
| Cross-validation F1 std | < 0.04 |

If asked anything that's not on this list, it's okay to say *"I'd have to check the notebook"*. Don't fabricate numbers.

---

## Mental Model #13 — The Defense Arc

The 12-minute presentation has a narrative arc. Internalize the shape:

1. **Set up the problem** (slides 1-3) → "Why does this matter?"
2. **Show the methodology** (slides 4-7) → "What did you build, and why those choices?"
3. **Land the key insight** (slide 8) → "The threshold-0.30 calibration is the moment of methodological maturity."
4. **Prove interpretability** (slide 9) → "Unlike BERT, every prediction is auditable."
5. **Demonstrate engineering** (slides 10-12) → "It's not just a model, it's a system."
6. **Live demo** (slide 13) → "Watch it work."
7. **Be honest about limits** (slide 14) → "Here's what it doesn't do."
8. **Close strong** (slide 15) → "Three takeaways."

Each slide should feel like it earns its place. If the prof loses interest at any point, the arc is broken.

---

## Mental Model #14 — How to Handle "I Don't Know"

You don't have to know everything. The prof will respect honesty more than fabrication.

**Good responses:**
- *"I haven't tested that exact case, but my hypothesis would be… because of [property of model]."*
- *"That's a great question. We didn't measure that explicitly, but it's something we could explore in future work."*
- *"I'd need to check the notebook to give you an exact number, but I remember the order of magnitude was…"*
- *"I'm not sure I followed the question — could you rephrase it?"*

**Bad responses:**
- Fabricating a number ("uh, I think it was 0.94…")
- Pretending you considered something you didn't ("yeah, we tried that and it didn't work")
- Going silent for more than 5 seconds

If you genuinely don't know, **say so and pivot to what you do know**. That's the mark of a researcher, not a student who memorized.

---

## Final Checklist Before the Defense

Print these and tick them off:

- [ ] Can you say the elevator pitch (Mental Model #1) without looking at notes?
- [ ] Can you explain why TF-IDF beat LSA in 30 seconds?
- [ ] Can you justify SVM over LR in one sentence?
- [ ] Can you explain why threshold 0.30 and not 0.50?
- [ ] Can you recall the 5 deployed numbers (F1, AUC, P, R, latency)?
- [ ] Do you have the live demo flow rehearsed end-to-end?
- [ ] Have you rehearsed handoffs with Zakaria?
- [ ] Can you handle "I don't know" gracefully?
- [ ] Have you read PRESENTATION_SCRIPT.md twice?
- [ ] Have you listened to the NotebookLM Audio Overview?

If all 10 boxes are ticked, **you are over-prepared**. Go in calm.

---

## One Last Thought

The prof has graded hundreds of student projects. What distinguishes a good defense from a great one is **not how much you know** but **how you reason out loud**. Show the prof your thinking process. Say *"we chose X because..."*. Say *"the alternative would have been Y, but..."*. Say *"this surprised us, so we looked into it and found..."*.

A student who can think on their feet beats a student who memorized perfectly.

You've got this. 🍀
