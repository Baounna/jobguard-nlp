# Screenshots Folder

Drop your live-demo screenshots here. Suggested filenames so the slide deck and the LaTeX report can reference them consistently:

| Filename | What to capture |
|---|---|
| `01_hero.png` | Full landing page — status bar, shield, stat strip, Try Sample buttons |
| `02_fraud_verdict.png` | After clicking **Try Fraud Sample** → **Analyze Posting** — the red 🚨 verdict card |
| `03_legit_verdict.png` | After clicking **Try Legit Sample** → **Analyze Posting** — the green ✅ verdict card |
| `04_topic_detection.png` | After pasting a non-job text like `CHAWARMA` — the orange "Not a Job Posting" card with topic icon |
| `05_score_breakdown.png` | Close-up of the Score Breakdown card with the gradient probability bar |
| `06_ocr_upload.png` *(optional)* | Image upload area with extracted-text preview |
| `07_signals.png` *(optional)* | Detected Warning Signs / Positive Signals tag row |

## How to capture (macOS)

| Type | Shortcut |
|---|---|
| Whole screen | `Cmd + Shift + 3` |
| Selected rectangle | `Cmd + Shift + 4`, drag |
| Specific window | `Cmd + Shift + 4`, then `Space`, click |

Screenshots save to your Desktop by default — drag them into this folder and rename them per the table above.

## How to use them

### In `SLIDES.md`
```markdown
![Live demo — fraud verdict](screenshots/02_fraud_verdict.png)
```

### In `FINAL_REPORT.tex`
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{screenshots/02_fraud_verdict.png}
\caption{Streamlit app showing a fraudulent posting verdict.}
\end{figure}
```

Then recompile: `pdflatex FINAL_REPORT.tex` (twice for cross-references).

### In NotebookLM
Upload them as additional sources alongside `FINAL_REPORT.pdf` so the Video Overview can reference real interface visuals.
