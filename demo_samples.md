# 🎤 Live Demo — Cheat Sheet for Oral Presentation

All 7 samples below have been **tested against the actual deployed model** (`models/svm_model.pkl`)
**at the deployed threshold of 0.30** (chosen via precision/recall sweep in notebook 4 §8).
Predictions and confidence scores are reproducible — no surprises in front of the prof.

---

## Recommended demo order

Run them in this order. Each takes ~10 seconds: paste → click **Analyze Posting →** → narrate the verdict.

| # | Type | Sample | Expected | Confidence |
|---|------|--------|----------|------------|
| 1 | Legit | Backend Engineer (Stripe) | ✅ LEGITIMATE | **98.7%** |
| 2 | Legit | Marketing Coordinator (Patagonia) | ✅ LEGITIMATE | **99.6%** |
| 3 | Legit | Financial Analyst (Deloitte) | ✅ LEGITIMATE | **98.4%** |
| 4 | Fraud | Data Entry / "earn $25/hr" | 🚨 FRAUDULENT | **99.9%** |
| 5 | Fraud | "Earn $5000/week from home" | 🚨 FRAUDULENT | **98.5%** |
| 6 | Fraud | Office Manager / reshipping | 🚨 FRAUDULENT | **97.4%** |
| 7 | **Borderline** | Subtle MLM ("Brand Partner") | 🚨 FRAUDULENT | **43.9% fraud** (caught only thanks to threshold = 0.30) |

> Sample #7 is the **threshold-tuning showcase**: at the default 0.5 threshold this MLM scam would
> slip through as "legit" (56%/44%). Our **tuned threshold of 0.30** catches it. Use this to introduce
> the precision/recall trade-off discussion. (See script at the bottom.)

---

## Sample #1 — Backend Engineer (LEGITIMATE)

**Title:** `Backend Software Engineer`
**Company:** `Stripe`

**Description:**
```
Backend Software Engineer — Stripe — Remote (EU)

We are looking for a Backend Software Engineer to join our Payments Reliability team.
You will design, build, and operate distributed systems that process millions of
transactions per day.

Responsibilities
- Design and implement scalable, fault-tolerant backend services in Go or Ruby
- Own services from prototype to production, including monitoring and on-call
- Collaborate on architecture reviews and write technical RFCs
- Mentor junior engineers and contribute to code reviews
```

**Requirements:**
```
- 4+ years of professional software engineering experience
- Strong knowledge of distributed systems and databases (PostgreSQL, Redis)
- Experience with at least one of: Go, Ruby, Java, Python
- Bachelor's degree in Computer Science or equivalent practical experience
```

**Benefits:**
```
- Competitive salary, equity, and quarterly performance bonus
- Comprehensive health, dental, and vision insurance
- 25 days paid vacation, parental leave, and learning stipend
```

**What to say:** *"A typical real-world tech posting — clear company, specific stack, structured benefits. The model is 98.7% confident this is legitimate."*

---

## Sample #2 — Marketing Coordinator (LEGITIMATE)

**Title:** `Marketing Coordinator`
**Company:** `Patagonia`

**Description:**
```
Marketing Coordinator — Patagonia — Reno, NV

Patagonia is seeking a Marketing Coordinator to support the execution of integrated
campaigns across digital, retail, and event channels.

What you will do
- Coordinate creative briefs, asset delivery, and timelines across teams
- Maintain marketing calendars and project trackers in Asana
- Partner with the social media and PR teams on launch moments
- Track campaign KPIs and prepare post-mortem reports
```

**Requirements:**
```
- 2-4 years experience in marketing, brand, or agency coordination
- Excellent written communication; comfort with Excel and project tools
- Bachelor's degree in Marketing, Communications, or related field
```

**Benefits:**
```
- Salary range: $62,000-$74,000 + bonus
- Medical, dental, vision, 401(k) match
```

**What to say:** *"A non-tech, real-world posting. Specific salary range, recognised brand, professional tone. 99.6% legit — the highest confidence score."*

---

## Sample #3 — Financial Analyst (LEGITIMATE)

**Title:** `Financial Analyst`
**Company:** `Deloitte`

**Description:**
```
Deloitte's Financial Advisory practice is hiring a Financial Analyst to support
M&A due diligence engagements for clients across the EMEA region.

Key responsibilities
- Build three-statement financial models and valuation analyses
- Conduct industry research and benchmarking
- Prepare client deliverables including reports and presentations
- Work closely with senior managers on cross-border transactions
```

**Requirements:**
```
- Bachelor's or Master's degree in Finance, Accounting, or Economics
- 1-3 years of experience in audit, investment banking, or consulting
- Proficient in Excel, PowerPoint, and financial modeling
- Fluent in English and French; Arabic a plus
```

**Benefits:**
```
- Structured training program and CFA sponsorship
- Health insurance, transport allowance, performance bonus
```

**What to say:** *"A Casablanca-based posting — relevant to our region. 98.4% legit."*

---

## Sample #4 — Data Entry Scam (FRAUDULENT)

**Title:** `Data Entry Clerk`
**Company:** *(leave empty — the absence of a company name is itself a fraud signal)*

**Description:**
```
Data Entry Clerk - Work From Home - Immediate Start

We are urgently hiring data entry clerks to work from home. No experience required -
full training provided.

Job duties:
- Typing data from scanned documents
- Basic data entry into our online system
- Flexible hours - work whenever you want

Compensation: Earn $25 per hour. Weekly direct deposit. Signing bonus paid after
first week.

To apply send your name, address, phone number and bank details to
hr@global-dataentry.work
```

**Requirements:**
```
- High school diploma
- Computer with internet connection
- Available to start immediately
```

**What to say:** *"This triggers the model's strongest fraud signals: `data entry`, `work home`, `signing bonus`, `typing data`, `earn` — exactly the bigrams the SVM learned from EMSCAD. **99.9% fraud.**"*

---

## Sample #5 — Earn From Home (FRAUDULENT)

**Title:** `Work From Home Opportunity`
**Company:** *(leave empty)*

**Description:**
```
Earn $5000 Per Week Working From Home!

Amazing opportunity! Work from home, no experience needed. Earn up to $5000 per week
guaranteed.

We are looking for motivated individuals to join our team. No interview required -
simply send us your personal details and start earning today.

What we offer:
- Unlimited earnings every week
- Work from anywhere
- No experience needed
- Get started immediately
```

**Benefits:**
```
- Unlimited earnings!
```

**What to say:** *"The textbook example. Vague salary promise, urgency, no requirements, no interview, asks for personal details upfront. 98.5% fraud."*

---

## Sample #6 — Office Manager / Reshipping (FRAUDULENT)

**Title:** `Office Manager - Part Time`
**Company:** `International Logistics`

**Description:**
```
International company is hiring an office manager to work from home. No previous
experience required.

Responsibilities:
- Process incoming payments
- Forward documents to our overseas team
- Receive packages on behalf of the company

Requirements:
- Must have a home address (no PO boxes)
- Available 2-3 hours per day
- Background check not required

Earn $3000 per month plus signing bonus. Get started immediately - send us your
name, address, and bank account info to begin processing payments this week.
```

**What to say:** *"This is a **real fraud archetype** — reshipping / payment-handler scams used in money-laundering schemes. 97.4% fraud."*

---

## Sample #7 — Subtle MLM (BORDERLINE — caught by threshold tuning) ⚠️

**Title:** `Brand Partner`
**Company:** *(leave empty)*

**Description:**
```
Brand Partner Opportunity - Health & Wellness

Are you tired of the 9-to-5? Looking for financial freedom?

We are a fast-growing health and wellness company expanding into your region. As a
Brand Partner you will promote our award-winning nutrition products, build your own
team, and earn residual income.

No selling experience required. We provide the training. Many partners earn six
figures within their first year.

Small investment required to get started. Contact Joey - limited spots.
```

**What to say:** *"This is an MLM-style fraud (multi-level marketing). The model is hesitant — only **44% fraud confidence** vs **56% legit**. At the default Streamlit threshold of 0.5, this scam would slip through. **But our deployed threshold is 0.30**, chosen by sweeping the precision/recall curve in notebook 4 §8 — and that's what catches this case. The lesson: in fraud detection, the default 0.5 threshold is the wrong default. For an even subtler MLM that scored under 30%, we'd still miss it — that's where BERT future-work comes in."*

---

## Suggested presentation flow (5-minute live demo)

1. **Open the app** — `streamlit run app.py` — show the dark UI, point out the sidebar pipeline
2. **Click "Test with Legit Sample"** in sidebar (built-in) → run → ✅
3. **Click "Test with Fraud Sample"** in sidebar (built-in) → run → 🚨
4. **Paste Sample #1 (Backend Engineer)** → real-world legit
5. **Paste Sample #4 (Data Entry)** → real-world fraud, 99.9%
6. **(Optional) Upload a real LinkedIn screenshot** — see "Screenshots" section below
7. **Paste Sample #7 (Subtle MLM)** → model is uncertain → bridge into your "Limitations" slide

---

## 📷 Screenshots & images for the slide deck

### Already generated by your notebooks (`data/processed/*.png`)

Use these in your slides — they prove your work, not just the conclusion.

| File | Use as slide |
|------|--------------|
| `class_distribution.png` | "Dataset is heavily imbalanced (4.84% fraud)" |
| `missing_values.png` | "Missing-data analysis motivates fillna with empty string" |
| `text_length_distribution.png` | "Fraudulent postings are shorter — descriptive feature" |
| `wordclouds.png` | Side-by-side legit vs fraud word clouds — **most striking visual** |
| `top_words.png` | Top discriminative words per class |
| `tfidf_top_features.png` | Highest-weight TF-IDF features overall |
| `tfidf_chi2_features.png` | Top discriminative features by Chi² test |
| `tsne_lsa.png` | t-SNE clustering — shows the two classes are separable |
| `confusion_matrices.png` | 2×2 grid for all 4 model variants |
| `roc_curves.png` | ROC overlay — proves SVM+TF-IDF dominates |
| `model_comparison.png` | Bar chart of 5 metrics × 4 models |
| `svm_feature_importance.png` | Top fraud / legit indicators learned by SVM |
| `threshold_tuning.png` | Precision-recall curve + F1 vs threshold — justifies operating point 0.30 |

### Real LinkedIn / Indeed screenshots (you take these yourself)

For the live demo's image-upload feature, search any of these and screenshot a posting:

**Legitimate (use any):**
- LinkedIn → search "software engineer Casablanca"
- Indeed → "marketing manager remote"
- Glassdoor → any major company you recognise (Google, OCP, Inwi, Maroc Telecom)

**Fraudulent (search terms that surface scams on real platforms):**
- LinkedIn → "data entry from home" — many are scams
- Facebook Marketplace / Jobs → "earn from home weekly"
- Telegram channels — `@JobsRemote` and similar are full of scam reposts

Save the screenshots to a `demo_screenshots/` folder, then drag them into the **Upload a Screenshot** widget during the live demo. The OCR will pull the text out automatically.

> **Tip:** For each screenshot, run the prediction *before* the demo and write the expected verdict on a sticky note. Don't get surprised live.

### Free stock images (if you need a title slide)

- **unsplash.com** — search "fraud", "scam", "warning", "remote work", "phishing"
- **undraw.co** — clean SVG illustrations matching your dark theme; search "security", "warning", "data"

---

## 🧠 Topic detection demo (out-of-domain inputs)

The app has a **scope gate** that detects when input isn't a job posting. Try these:

| Paste this | Expected card |
|---|---|
| `CHAWARMA` | ❌ Not a job posting → 🍔 **Food / Restaurant / Delivery** |
| `Bitcoin and Ethereum prices crashed today` | ❌ Not a job posting → ₿ **Cryptocurrency / Blockchain** |
| `Real Madrid won the Champions League final` | ❌ Not a job posting → ⚽ **Sports** |
| `Take a deep breath and relax in this yoga session` | ❌ Not a job posting → 🧘 **Wellness / Yoga / Meditation** |
| `Subscribe to my YouTube and follow me on TikTok` | ❌ Not a job posting → 📱 **Social Media** |
| `My grandma took the baby to kindergarten` | ❌ Not a job posting → 👨‍👩‍👧 **Family / Parenting** |
| `Preheat oven, mix flour, sugar, eggs` | ❌ Not a job posting → 👨‍🍳 **Cooking / Recipes** |
| `Hurricane caused massive flooding, heatwave records broken` | ❌ Not a job posting → ⛅ **Weather / Climate** |
| `asdf qwer zxcv hjkl mnbv` (gibberish) | ❌ Not a job posting → 🧠 **Out of model's memory** (none of 39 topics matched) |

**Talking points:**
- 39 topics stored in memory.
- The system **refuses to guess** when no topic matches — honest UX.
- The card shows alternative plausible topics with their match counts when relevant.

---

## 💬 Q&A defenses (likely prof questions)

| Question | Short answer |
|----------|--------------|
| *Why TF-IDF and not BoW?* | "BoW is competitive on F1 (0.82 vs 0.79) but TF-IDF wins ROC-AUC (0.986 vs 0.980) — better-calibrated probabilities for the demo's confidence display. Notebook 3 has the side-by-side." |
| *Why SVM over Logistic Regression?* | "SVM has higher precision (97.8% vs 70.4%), so fewer false positives. LR has higher recall. We chose SVM because in a **scam detector**, falsely accusing a legit company is worse than the user double-checking. We discuss this trade-off in the report." |
| *Why not BERT?* | "Assignment forbids transformers in the main pipeline. We discuss BERT as a future direction — exactly because of the MLM case the model just missed." |
| *How do you handle the 4.84% imbalance?* | "`class_weight='balanced'` in both LinearSVC and LogisticRegression. We also report F1 and ROC-AUC, not accuracy." |
| *Could the model be fooled?* | "Yes — Sample #7 demonstrated it. Sophisticated scams that mimic legitimate business vocabulary evade keyword-based features. This is a known limitation of bag-of-words representations." |
| *What's the dataset size?* | "17,880 postings, 866 fraud, ~5%. EMSCAD via Kaggle." |
