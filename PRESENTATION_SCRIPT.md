# 🎤 Oral Presentation Script

**Project:** Job Offer Scam Detection — A Classical NLP Pipeline
**Presenters:** Mohamed Baounna · Zakaria Birani
**Audience:** Pr. H. El Hamdaoui — NLP Unit, SIIA Semester 6, FP Khouribga
**Total time:** ~12 minutes presentation + 3–5 minute live demo

---

## How to use this script

- **[M]** = Mohamed says this · **[Z]** = Zakaria says this
- **[Action]** = something to do (click, point, demo)
- **[Pause]** = brief pause for emphasis or transition
- Don't read it word-for-word — *internalize the structure* and let the words flow naturally
- **Rehearse the script twice** before the oral, once silently, once out loud
- If you're more comfortable in French, translate it the night before — the structure stays the same

---

## 🎬 Opening — Slide 1 (Title)

**[Both stand at the front. Mohamed clicks the slide.]**

**[M]** *Bonjour Professeur. Good morning.*

**[M]** I'm Mohamed Baounna, this is my partner Zakaria Birani, and today we'll present our final NLP project: a system that automatically detects fraudulent job postings using classical machine learning.

**[M]** Our model achieves an F1-score of 0.892 on the EMSCAD dataset, and we deployed it as an interactive web application called **JobGuard**.

**[Pause — click to next slide]**

---

## 🚨 Slide 2 — The Scam Epidemic & The Metric Trap

**[Z]** Online job boards like LinkedIn or Indeed host thousands of postings — but a real fraction of them are scams. They cost real people their identity, their money, sometimes their whole career path.

**[Z]** The dataset we used is called **EMSCAD**. It contains **17,880 real job postings**, of which only **4.84% are fraudulent**. This means a model that always predicts "legitimate" would still be 95% accurate — and completely useless.

**[Z]** That's why we report **F1-score and ROC-AUC** as our primary metrics, not accuracy.

**[Pause — click]**

---

## 🔤 Slide 3 — Two Distinct Vocabularies

**[M]** Before choosing any model, we looked at the **vocabulary**.

**[M]** *(point to the green word cloud)* Legitimate postings use professional words: `team`, `client`, `experience`, `customer`.

**[M]** *(point to the red word cloud)* Fraudulent postings use a completely different lexicon: `earn`, `urgent`, `unlimited`, `home`, `free`. They're also about 50 tokens shorter on average.

**[M]** This vocabulary divergence is the foundation of our project. It tells us classical methods like Bag-of-Words and TF-IDF will naturally capture the fraud signal — we don't need a neural network for this.

**[Pause — click]**

---

## 🏗️ Slide 4 — The Architecture

**[Z]** Our pipeline follows the **six steps required by the assignment**: Data Assembly, Preprocessing, Representation, Model Training, Threshold Calibration, and Deployment.

**[Z]** We respected one critical constraint: **no transformers in the main pipeline**. The deployed model is purely classical machine learning. BERT is mentioned only as future work, never as the deployed solution.

**[Pause — click]**

---

## 🧹 Slide 5 — Preprocessing with Purpose

**[M]** Each preprocessing step has a clear justification.

**[M]** We **combine the five text fields** — title, company, description, requirements, benefits — into one document. We **lowercase**, **remove URLs and special characters with regex**, **strip stop-words** (which removes about 30% of the vocabulary), and finally we **lemmatize** with NLTK's WordNetLemmatizer.

**[M]** *(point to the right panel)* We deliberately chose **lemmatization over stemming**. Lemmatization preserves real, readable English words, which means our final feature-importance plots are auditable and human-readable. That's an engineering choice, not a default.

**[Pause — click]**

---

## 🔢 Slide 6 — The Representation Matrix

**[Z]** The assignment asks us to compare at least two text representations and justify our choice. We compared three:

**[Z]** **Bag-of-Words** with 15,000 features reaches an F1-score of **0.825**.

**[Z]** **TF-IDF** with the same vocabulary reaches **0.793 on F1, but the highest ROC-AUC at 0.986** — meaning it produces the best-calibrated probability rankings.

**[Z]** **LSA**, a dense semantic representation, drops dramatically to **0.381**.

**[Z]** *(point to the explanation at the bottom)* **Why does LSA fail?** Because the SVD projection compresses out the rare, highly discriminative scam tokens — exactly the words we need to detect fraud. **TF-IDF wins because its inverse-document-frequency weighting amplifies those rare signals**.

**[Z]** That's why TF-IDF is our chosen representation.

**[Pause — click]**

---

## ⚖️ Slide 7 — Algorithm Face-Off

**[M]** Same logic for the algorithms — we compared **Linear SVM** and **Logistic Regression**, both on TF-IDF and on LSA.

**[M]** *(point to the bar chart)* Logistic Regression wins on raw recall — 90.8% — meaning it catches more scams. But it has poor precision: 70.4%, which means too many legitimate jobs are wrongly flagged as fraud.

**[M]** **Linear SVM**, calibrated with `CalibratedClassifierCV`, wins decisively on precision: **97.8%**.

**[M]** *(point to the verdict box)* **We deployed the Calibrated Linear SVM**. In a real-world fraud detector, falsely accusing a legitimate company is much more costly than missing one scam. High precision is the baseline requirement.

**[Pause — click]**

---

## 🎚️ Slide 8 — The Operating Point: Threshold Calibration

**[Z]** This is one of the most important slides.

**[Z]** Most students stop at the default 0.5 decision threshold. But for an imbalanced dataset, **0.5 is a mathematical illusion**, not a business decision.

**[Z]** *(point to the right column)* At 0.5, our SVM has Precision 0.978, Recall 0.775, F1 0.865.

**[Z]** We swept the threshold across the precision-recall curve and found **the optimal point at 0.30**: Precision 0.931, Recall 0.855, **F1 = 0.892**.

**[Z]** **We trade 5 points of precision to rescue 8 points of recall.** In fraud detection, missing a scam costs the user far more than flagging a borderline posting for a manual second look.

**[Z]** Threshold calibration alone — without changing the model at all — lifted our F1 by almost 3 percentage points.

**[Pause — click]**

---

## 🔍 Slide 9 — Inside the Black Box: Total Interpretability

**[M]** Because we're using a linear SVM on TF-IDF features, **every prediction is fully traceable**.

**[M]** *(point to the bars)* On the left in red, the **Scammer's Lexicon** — bigrams that increase the fraud score: *data entry, work home, signing bonus, typing data*.

**[M]** On the right in green, the **Corporate Lexicon** — *client, recruitment, interview, team*.

**[M]** This is a major advantage of classical ML over deep models. Unlike a transformer, where we can't explain why a specific prediction was made, **JobGuard can show the exact features that drove every decision**. This is critical for any auditing or regulatory context.

**[Pause — click]**

---

## 💻 Slide 10 — Deployment Reality: Meet JobGuard

**[Z]** All of this lives inside an interactive Streamlit web application called **JobGuard**.

**[Z]** *(point to the annotations on the screenshot)* It has live system telemetry at the top, one-click sample buttons for instant testing, and a screenshot drop zone for multimodal input.

**[Z]** The full deployed model — the SVM and the TF-IDF vectorizer — is **less than 1 megabyte** and inference takes **5 milliseconds per posting**. That's a real engineering advantage over a BERT pipeline, which would be 90 megabytes and 10 times slower.

**[Pause — click]**

---

## 📷 Slide 11 — The Multimodal Edge: OCR Integration

**[M]** Beyond text input, JobGuard can accept **screenshots**.

**[M]** When a user drags a LinkedIn or Indeed screenshot, **Tesseract OCR** extracts the text in the background and auto-fills the description field. We cache the result by file ID, so the OCR runs only once per upload — and we never overwrite manual edits.

**[M]** This reduces friction for users and lets us catch scams that try to hide behind non-selectable image text.

**[Pause — click]**

---

## 🛡️ Slide 12 — The Scope Gate: Refusing to Hallucinate

**[Z]** A naive classifier trained only on job boards would confidently label a cake recipe or a login page as "Legitimate Job" — simply because such inputs lack fraud-specific words.

**[Z]** We solved this with a **scope gate**. Before the SVM runs, we count how many job-related lemmas appear in the input — out of about 50 known terms. If we see at least 3, the input goes to the SVM. If not, it's routed to a **39-topic fallback detector** which identifies whether the text is about food, sports, login pages, news, and so on.

**[Z]** And if the input doesn't match any known topic, JobGuard **explicitly refuses to guess** and tells the user. This is honest UX — better than a confident wrong answer.

**[Pause — click]**

---

## 🎯 Slide 13 — Live Audit: Three Case Studies

**[M]** Let's see all of this work in real time. *(turn to the screen with the live app open)*

**[M]** Three test cases — easy, easy, and hard.

### 🟢 Test 1 — The Obvious Legit

**[M]** *(click "Try Legit Sample")* This is a Backend Engineer role at TechCorp — clear stack, structured benefits.

**[M]** *(click "Analyze Posting")* **98.7% Legitimate.** The system shows positive signals: *Clear job description, Company info present, Specific requirements*. ✅

### 🔴 Test 2 — The Obvious Scam

**[M]** *(click "Try Fraud Sample")* "Earn $5000/week from home, no company, fake urgency."

**[M]** *(click "Analyze Posting")* **99.9% Fraudulent.** Warning signs: *Vague salary promise, Urgency tactics, Missing company info*. 🚨

### 🟡 Test 3 — The Borderline MLM

**[M]** *(paste a Brand Partner / Multi-Level-Marketing pitch)* This is a much subtler scam. It mimics legitimate business vocabulary: *brand, team, training, opportunity*.

**[M]** *(click "Analyze Posting")* **Verdict: 43.9% Fraudulent — caught.**

**[M]** And here's the key insight: **at the default 0.5 threshold, this MLM scam would slip through as legitimate**. It only gets caught because we deployed at the **engineered 0.30 threshold**. This is exactly the value of our Slide 8 calibration — translated into a real save.

**[Pause — click]**

---

## 🔮 Slide 14 — Error Analysis & The Horizon

**[Z]** We're honest about the limitations.

**[Z]** **First**, about 22% of the most subtle MLM scams are still missed. The fix is **transformer integration** — BERT-style contextual embeddings could capture the phrase-level semantics that bag-of-words cannot.

**[Z]** **Second**, our vocabulary is English-only. The fix is **multilingual training** — incorporating Arabic and French EMSCAD subsets to serve markets like Casablanca.

**[Z]** **Third**, our scope gate uses a fixed lexicon, which is fragile. The fix is a **zero-shot upgrade** with a model like `bart-large-mnli` for richer semantic topic recognition.

**[Z]** The architecture was designed to accept these upgrades modularly without rewriting the core.

**[Pause — click]**

---

## 🏁 Slide 15 — The Final Verdict

**[M]** Three takeaways from this project:

**[M]** **First — classical ML endures.** Bag-of-words and linear SVMs remain highly competitive for semantic classification, especially when the discriminative signal is in rare vocabulary.

**[M]** **Second — thresholds require business logic.** The default 0.5 is mathematical, not practical. Threshold calibration is non-optional.

**[M]** **Third — interpretability is a feature, not a bug.** In auditing and security contexts, transparent feature weighting outperforms opaque deep learning.

**[M]** **Final performance: F1 = 0.892, latency = 5 milliseconds, model size under 1 megabyte.**

**[Both]** *Thank you, Professor. We're ready for your questions.* 🙏

---

# 🎙️ Anticipated Q&A — short defenses

If the prof asks any of these, here's the one-line answer:

| Question | Short answer |
|---|---|
| *Why TF-IDF and not BoW?* | "BoW is competitive on F1, but TF-IDF wins on ROC-AUC, which means better-calibrated probabilities for the demo's confidence scores." |
| *Why SVM over Logistic Regression?* | "Higher precision — 97.8% vs 70.4%. In a scam detector, falsely accusing a legit company is worse than missing one scam." |
| *Why not BERT?* | "Assignment forbids transformers in the main pipeline. We propose BERT as future work for the MLM cases we miss." |
| *Why threshold 0.30 specifically?* | "F1-optimal point on the precision-recall curve. Lifts F1 from 0.865 to 0.892 with no model change." |
| *How did you handle the 4.8% imbalance?* | "`class_weight='balanced'` in both SVM and LR, plus threshold tuning, plus reporting F1 and ROC-AUC instead of raw accuracy." |
| *Could the model be tricked?* | "Yes — Slide 13's MLM case showed it. Sophisticated scams that mimic legitimate vocabulary evade keyword-based features. That's the BERT future-work motivation." |
| *Is 17,880 postings enough?* | "Reasonable for classical ML. The model isn't overfitting — train/test gap is small. For a deeper model, more data would help." |
| *Why no LSTM or CNN?* | "Same constraint — no neural networks in the main pipeline. Plus, our feature-importance analysis would be impossible with a recurrent or convolutional model." |
| *Did you do cross-validation?* | "Yes — 5-fold stratified CV on the training set only. F1 standard deviations are below 0.03, so the performance is stable across folds, not a property of a lucky split. Details in report §4.4." |
| *Why default hyperparameters (C=1.0)?* | "We chose defaults because the ROC-AUC was already 0.98+ on cross-validation. Grid search around C ∈ {0.1, 1, 10} produced changes under 1 percentage point of F1 — not worth the added complexity. Documented in our methodology." |
| *Why not balance the dataset with SMOTE or undersampling?* | "We used `class_weight='balanced'` which gives a similar effect without inventing synthetic data. SMOTE can blur the boundary between rare and common scam vocabulary, which is exactly the signal we want preserved. Documented in §6 limitations." |

---

# ⏱️ Timing reference

| Section | Slides | Approx. time |
|---|---|---|
| Opening + problem framing | 1–2 | 1 min 30 |
| Methodology | 3–7 | 4 min 30 |
| Threshold calibration (key moment) | 8 | 1 min |
| Engineering & deployment | 9–12 | 2 min 30 |
| Live demo | 13 | 3–4 min |
| Limitations & conclusion | 14–15 | 1 min 30 |
| **Total** | **15** | **~14–15 min** |

---

# 💡 Last-minute tips

1. **Speak slowly.** Most students rush. Aim for ~140 words per minute.
2. **Look at the prof, not the screen.** Glance at the slide, then turn back.
3. **Hand off cleanly.** When switching from Mohamed to Zakaria (or vice versa), end your sentence and step back; let your partner step forward. Never both speak at once.
4. **The live demo is your strongest moment.** Practice it 5+ times before the oral. If the Wi-Fi fails, have the deck PDF ready as backup.
5. **The MLM case (Slide 13, Test 3) is your "wow moment."** Land that line slowly: *"At the default threshold this scam would slip through. It only gets caught because we engineered the threshold to 0.30."* That's the sentence the prof will remember.
6. **End with confidence.** "Thank you, we're ready for your questions" — pause, smile, look at the prof.

Bonne chance! 🍀
