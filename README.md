# Emotion-Aware Sarcasm Detection System

> **Minor Project (EGR4501)** | Department of Computer Science & Engineering  
> School of Engineering and Sciences | GD Goenka University, Gurugram  
> Academic Session 2026-27

## Team

| Name              | Roll Number      |
|-------------------|------------------|
| Soham Bhole       | 230160226004     |
| Gaurav Kumar      | 230160226009     |
| Rishit Choudhary  | 230160223113     |
| Mohd Afnan        | 230160203202     |

**Supervisor:** Dr Yogesh Kumar

---

## Overview

A text-based research prototype that predicts whether English text is **sarcastic** or **non-sarcastic**, using emotion as an auxiliary signal. The system compares a **text-only baseline** (TF-IDF + Logistic Regression) with an **emotion-aware model** (TF-IDF + emotion features + Logistic Regression) to test whether emotion information improves classification.

## Quick Start

### Prerequisites
- Python 3.9 or later
- pip (Python package installer)
- Internet connection (for first-time dataset and model download)

### Setup and Run

```bash
# 1. Navigate to the project directory
cd emotion-sarcasm-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the models (downloads dataset + computes emotion features + trains)
python train.py                 # Full training (~15-20 min on CPU)
# OR
python train.py --quick         # Quick mode with 3000 samples (~3 min)
# OR
python train.py --no-transformer  # Use lexicon-based emotion (no model download, instant)

# 5. Start the web application
python app.py

# 6. Open in browser
# http://localhost:5000
```

### Windows Quick Start
Simply double-click `run.bat` to install dependencies, train models (quick mode) and start the server.

---

## Project Structure

```
emotion-sarcasm-detection/
├── app.py                      # Flask web application
├── train.py                    # Training and evaluation pipeline
├── config.json                 # Configuration parameters
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── run.bat                     # Windows convenience script
│
├── data/                       # Dataset module
│   ├── __init__.py
│   ├── dataset.py              # Download, load and split dataset
│   ├── raw_data.json           # (Downloaded) Raw sarcasm dataset
│   ├── train.csv               # (Generated) Training split
│   ├── val.csv                 # (Generated) Validation split
│   └── test.csv                # (Generated) Test split
│
├── models/                     # ML model modules
│   ├── __init__.py
│   ├── preprocessor.py         # Text preprocessing
│   ├── emotion_classifier.py   # Emotion analysis (transformer + lexicon)
│   └── sarcasm_detector.py     # Baseline and emotion-aware classifiers
│
├── evaluation/                 # Evaluation module
│   ├── __init__.py
│   └── evaluator.py            # Metrics, comparison and error analysis
│
├── saved_models/               # (Generated) Trained model files
├── results/                    # (Generated) Evaluation results
│
├── templates/
│   └── index.html              # Web interface template
│
└── static/
    ├── css/style.css           # Stylesheet
    └── js/main.js              # Client-side JavaScript
```

## Architecture

```
Browser ──► Flask API ──► Preprocessing ──► Emotion Classifier ──┐
                                         └──► Sarcasm Classifier ◄─┘
                                              (Baseline / Emotion-Aware)
                                                      │
                                              Results Formatter ──► Response
```

- **Text-Only Baseline:** TF-IDF (10,000 features, unigrams + bigrams) → Logistic Regression
- **Emotion-Aware Model:** TF-IDF features + 7 emotion probabilities → Logistic Regression (late fusion)
- **Emotion Classifier:** Pretrained `distilroberta` (7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise)

## Dataset

**News Headlines Dataset for Sarcasm Detection v2** (Misra, 2019)  
- ~28,600 news headlines labelled as sarcastic or non-sarcastic  
- Source: [GitHub](https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection)  
- Licence: CC BY-SA 4.0 (academic use)

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score, Macro-F1
- Confusion Matrix
- Inference Latency (ms per sample)

## API Endpoints

| Method | Endpoint    | Description                     |
|--------|-------------|---------------------------------|
| GET    | /           | Web interface                   |
| POST   | /analyze    | Analyse text (JSON body)        |
| GET    | /evaluate   | View evaluation results         |
| GET    | /health     | Health check                    |
| GET    | /api/info   | System information              |

## References

1. Joshi, A., Bhattacharyya, P., & Carman, M. J. (2017). Automatic Sarcasm Detection: A Survey. *ACM Computing Surveys*, 50(5), Article 73.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*, 4171-4186.
3. Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *Proceedings of ACL*, 4040-4054.
4. Misra, R. (2019). News Headlines Dataset for Sarcasm Detection. GitHub repository.

---

*This system is a research prototype. Predictions are automated estimates and should not be used for high-stakes decisions.*
