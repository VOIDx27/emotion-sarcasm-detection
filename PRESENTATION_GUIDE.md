# Phase-1 Project Presentation & Viva Defense Guide
**Minor Project (EGR4501) | Date: 24 October 2026**  
**Department of Computer Science & Engineering | GD Goenka University**  
**Project:** Emotion-Aware Sarcasm Detection System  
**Supervisor:** Dr Yogesh Kumar  
**Team:** Soham Bhole, Gaurav Kumar, Rishit Choudhary, Mohd Afnan

---

## 1. Executive Summary for Evaluators

- **Problem:** Sarcasm inverts literal semantics ("What a great day to get a flat tire"). Traditional sentiment models misclassify sarcastic text as positive because of surface lexicon.
- **Research Question:** Does adding fine-grained emotion representations improve sarcasm detection over a rigorous text-only baseline?
- **Dataset:** News Headlines Dataset for Sarcasm Detection v2 (Misra, 2019) containing 26,602 annotated headlines (14,951 non-sarcastic, 11,651 sarcastic).
- **Architecture:** 
  - Text-Only Baseline: Sublinear TF-IDF (10,000 max n-grams: unigrams + bigrams) + Balanced Logistic Regression.
  - Emotion Component: Pretrained Transformer (`j-hartmann/emotion-english-distilroberta-base`) extracting a 7-dimensional probability vector (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`).
  - Emotion-Aware Model: Late fusion concatenating TF-IDF (10,000 dims) and Emotion features (7 dims) into a 10,007-dimensional feature space fed into Balanced Logistic Regression.
- **Evaluation Discipline:** No data leakage. Train / Validation / Test split (70% / 15% / 15%) created before model training. Test set (3,991 samples) kept untouched until final evaluation.

---

## 2. Experimental Results (Held-Out Test Set: 3,991 Samples)

| Metric | Text-Only Baseline | Emotion-Aware Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 83.74% | **83.81%** | **+0.07%** |
| **Precision** | 78.91% | **79.15%** | **+0.24%** |
| **Recall** | 85.81% | 85.58% | -0.23% |
| **F1-Score** | 82.21% | **82.24%** | **+0.03%** |
| **Macro-F1** | 83.62% | **83.69%** | **+0.07%** |
| **Inference Latency** | ~0.01 ms / sample | ~0.01 ms / sample | Instantaneous |

### Confusion Matrix Breakdown (Test Set: 3,991 Samples)
- **Baseline:** True Negatives = 1842, False Positives = 401, False Negatives = 248, True Positives = 1500
- **Emotion-Aware:** True Negatives = 1849, False Positives = 394, False Negatives = 252, True Positives = 1496

---

## 3. How to Answer Supervisor & Panel Questions (Viva Q&A)

### Q1: "Why did emotion features only produce a slight improvement (+0.07% to +0.24% precision)?"
> **Answer:** "This is one of the key empirical findings of our Phase-1 research. In headline-style sarcasm, lexical incongruity and word choices already provide strong signals captured by TF-IDF n-grams (reaching 83.74% baseline). Furthermore, our emotion model was pretrained on general dialogue/social text, so headline domain mismatch introduces minor noise. As outlined in our SRS, our goal was to test this hypothesis empirically and report the results transparently rather than artificially claim an exaggerated improvement. In Phase-2, we plan to evaluate cross-domain datasets (e.g., Twitter/Reddit) where emotional polarity incongruity is more pronounced."

### Q2: "Did you leak any test data into your emotion feature extraction or vocabulary?"
> **Answer:** "No, we strictly enforced research integrity as specified in FR-5.5 of our SRS:
> 1. The train (18,620), validation (3,991), and test (3,991) splits were generated using stratified sampling before any feature fitting.
> 2. The TF-IDF vectorizer vocabulary was fitted **strictly on the training split**.
> 3. The test set was held out and evaluated only once with the frozen configuration."

### Q3: "How is emotion extracted during real-time inference on the web app?"
> **Answer:** "At runtime (FR-4.4), the user inputs an arbitrary sentence. The text passes through our cleaning pipeline (preserving capitalization and expressive punctuation cues). The text is fed into our DistilRoBERTa emotion pipeline to generate probabilities across the 7 emotion classes. These 7 values are appended to the user text's TF-IDF vector, and our late-fusion classifier outputs the sarcasm prediction and confidence score."

### Q4: "What ethical and practical limitations exist in this system?"
> **Answer:** "As required by NFR-6 and FR-6.2:
> 1. Predictions are labeled as automated estimates, never as verified statements of speaker intent.
> 2. No user-submitted text is stored to database or disk (ephemeral in-memory processing).
> 3. The system clearly states that it is not intended for high-stakes decisions or autonomous content moderation."

---

## 4. Live Demonstration Steps for the Panel

1. **Open the browser:** Navigate to `http://localhost:5000`.
2. **Show the System Details:**
   - Click the **"ℹ️ About"** button in the header. Shows the project title, team members (Soham, Gaurav, Rishit, Mohd Afnan), supervisor Dr Yogesh Kumar, and architecture.
3. **Show Held-Out Evaluation Results:**
   - Click **"📊 Evaluation Results"** in the header.
   - Highlights the side-by-side comparison tables, confusion matrices, and empirical verdict from the 3,991 held-out test samples.
4. **Demonstrate Live Inference:**
   - Click the first chip: *"What a great day to be stuck in traffic for three hours!"* -> Click **Analyse**. Shows high surprise/anger emotion scores and outputs **SARCASTIC** with confidence.
   - Click the second chip: *"The government announced a new funding package for public schools today."* -> Click **Analyse**. Shows neutral emotion and outputs **NON-SARCASTIC**.
   - Type a custom sentence from the panel to demonstrate live inference in under 50ms.
