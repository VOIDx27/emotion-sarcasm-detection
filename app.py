"""
Flask web application for the Emotion-Aware Sarcasm Detection System.

Provides a web interface and REST API for sarcasm analysis (FR-1 to FR-7).
Input validation, preprocessing, emotion analysis and sarcasm prediction
are integrated into a single request pipeline.

Routes:
    GET  /           - Web interface
    POST /analyze    - Analyse text for sarcasm and emotion
    GET  /evaluate   - View saved evaluation results
    GET  /health     - Health check endpoint
    GET  /api/info   - System information
"""

import json
import os
import sys
import time

from flask import Flask, render_template, request, jsonify

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.preprocessor import TextPreprocessor
from models.emotion_classifier import (
    create_emotion_classifier,
    get_feature_vector,
    EMOTION_LABELS,
)
from models.sarcasm_detector import BaselineSarcasmDetector, EmotionAwareSarcasmDetector

# ---- Application Setup ----
app = Flask(__name__)

# Load configuration
config_path = os.path.join(PROJECT_ROOT, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Initialise components
preprocessor = TextPreprocessor()
emotion_classifier = None
baseline_model = None
emotion_aware_model = None
models_loaded = False

LIMITATION_NOTICE = (
    "This is an automated prediction from a research prototype. "
    "The output is an estimate based on text features and should not be treated as fact. "
    "Sarcasm depends heavily on context, tone and shared knowledge that text alone may not capture. "
    "This system is not intended for high-stakes decisions."
)


def load_models():
    """Load all model components. Called on first request."""
    global emotion_classifier, baseline_model, emotion_aware_model, models_loaded

    if models_loaded:
        return True

    models_dir = os.path.join(PROJECT_ROOT, "saved_models")

    try:
        # Load emotion classifier
        print("[App] Loading emotion classifier...")
        emotion_classifier = create_emotion_classifier(config, use_transformer=True)

        # Load baseline model
        print("[App] Loading baseline sarcasm model...")
        baseline_model = BaselineSarcasmDetector(config)
        baseline_model.load(models_dir)

        # Load emotion-aware model
        print("[App] Loading emotion-aware sarcasm model...")
        emotion_aware_model = EmotionAwareSarcasmDetector(config)
        emotion_aware_model.load(models_dir)

        models_loaded = True
        print("[App] All models loaded successfully!")
        return True

    except Exception as e:
        print(f"[App] Error loading models: {e}")
        print("[App] Make sure you have run 'python train.py' first!")
        return False


# ---- Routes ----

@app.route("/")
def index():
    """Serve the main web interface."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyse text for sarcasm and emotion (UC-1, UC-2).

    Request body (JSON):
        text: str - The text to analyse.

    Returns (JSON):
        success, input_text, emotion, baseline, emotion_aware, limitation_notice
        or error message for invalid input (UC-3).
    """
    # Ensure models are loaded
    if not load_models():
        return jsonify({
            "success": False,
            "error": "Models are not loaded. Please run 'python train.py' first.",
        }), 503

    # Parse input
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({
            "success": False,
            "error": "Please provide text in the request body as JSON: {\"text\": \"your text\"}",
        }), 400

    raw_text = data["text"]

    # Validate input (FR-1.3, FR-1.4)
    is_valid, error_msg = preprocessor.validate_input(
        raw_text,
        min_length=config.get("min_input_length", 3),
        max_length=config.get("max_input_length", 500),
    )
    if not is_valid:
        return jsonify({"success": False, "error": error_msg}), 400

    try:
        # Preprocess
        cleaned_text = preprocessor.preprocess(raw_text)

        # Emotion analysis (FR-3.1, FR-3.2)
        start_time = time.time()
        emotion_result = emotion_classifier.predict(cleaned_text)
        emotion_time = time.time() - start_time

        # Emotion feature vector for the emotion-aware model
        emotion_features = get_feature_vector(emotion_result)

        # Baseline sarcasm prediction (FR-4.1)
        start_time = time.time()
        baseline_result = baseline_model.predict(cleaned_text)
        baseline_time = time.time() - start_time

        # Emotion-aware sarcasm prediction (FR-4.1, FR-4.4)
        start_time = time.time()
        emotion_aware_result = emotion_aware_model.predict(cleaned_text, emotion_features)
        emotion_aware_time = time.time() - start_time

        # Build response (FR-6.1, FR-6.2)
        response = {
            "success": True,
            "input_text": raw_text,
            "preprocessed_text": cleaned_text,
            "emotion": {
                "label": emotion_result["label"],
                "scores": emotion_result["scores"],
                "note": "Estimated emotion category, not a verified emotional state.",
                "inference_time_ms": round(emotion_time * 1000, 1),
            },
            "baseline": {
                "model_name": "Text-Only Baseline (TF-IDF + Logistic Regression)",
                "label": baseline_result["label"],
                "is_sarcastic": baseline_result["is_sarcastic"],
                "confidence": baseline_result["confidence"],
                "probabilities": baseline_result["probabilities"],
                "inference_time_ms": round(baseline_time * 1000, 1),
            },
            "emotion_aware": {
                "model_name": "Emotion-Aware Model (TF-IDF + Emotion + Logistic Regression)",
                "label": emotion_aware_result["label"],
                "is_sarcastic": emotion_aware_result["is_sarcastic"],
                "confidence": emotion_aware_result["confidence"],
                "probabilities": emotion_aware_result["probabilities"],
                "inference_time_ms": round(emotion_aware_time * 1000, 1),
            },
            "limitation_notice": LIMITATION_NOTICE,
        }

        return jsonify(response)

    except Exception as e:
        # FR-7.1: Handle unexpected errors gracefully
        return jsonify({
            "success": False,
            "error": f"An error occurred during analysis: {str(e)}",
        }), 500


@app.route("/evaluate", methods=["GET"])
def evaluate():
    """Return saved evaluation results (UC-4)."""
    results_path = os.path.join(PROJECT_ROOT, "results", "evaluation_results.json")

    if not os.path.exists(results_path):
        return jsonify({
            "success": False,
            "error": "No evaluation results found. Run 'python train.py' first.",
        }), 404

    with open(results_path, "r") as f:
        results = json.load(f)

    results["success"] = True
    return jsonify(results)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/info", methods=["GET"])
def api_info():
    """Return system information."""
    return jsonify({
        "name": "Emotion-Aware Sarcasm Detection System",
        "version": "1.0",
        "project": "EGR4501 Minor Project",
        "university": "GD Goenka University",
        "team": [
            "Soham Bhole (230160226004)",
            "Gaurav Kumar (230160226009)",
            "Rishit Choudhary (230160223113)",
            "Mohd Afnan (230160203202)",
        ],
        "supervisor": "Dr Yogesh Kumar",
        "emotion_labels": EMOTION_LABELS,
        "models_loaded": models_loaded,
    })


# ---- Error Handlers ----

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ---- Main ----

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Emotion-Aware Sarcasm Detection System")
    print("=" * 60)

    # Pre-load models on startup
    print("[App] Pre-loading models...")
    if load_models():
        print("[App] Ready! Starting web server...")
    else:
        print("[App] WARNING: Models could not be loaded.")
        print("[App] Run 'python train.py' first, then restart this app.")
        print("[App] Starting server anyway (API will return errors)...")

    port = config.get("flask_port", 5000)
    host = config.get("flask_host", "0.0.0.0")
    debug = config.get("flask_debug", False)

    print(f"[App] Open your browser at: http://localhost:{port}")
    print(f"[App] Press Ctrl+C to stop the server.\n")

    app.run(host=host, port=port, debug=debug)
