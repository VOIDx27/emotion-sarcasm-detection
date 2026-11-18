"""
Training script for the Emotion-Aware Sarcasm Detection System.

This script:
1. Downloads and prepares the sarcasm dataset
2. Creates reproducible train / validation / test splits
3. Computes emotion features for all data
4. Trains the text-only baseline model
5. Trains the emotion-aware model
6. Evaluates both on the same held-out test set
7. Saves models, results and error samples

Usage:
    python train.py              # Full training (~15 min with transformer)
    python train.py --quick      # Quick mode with 3000 samples (~2 min)
    python train.py --no-transformer  # Use lexicon-based emotion (instant)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import DatasetManager
from models.preprocessor import TextPreprocessor
from models.emotion_classifier import (
    create_emotion_classifier,
    get_feature_vectors,
    cache_emotion_features,
    load_emotion_cache,
    EMOTION_LABELS,
)
from models.sarcasm_detector import BaselineSarcasmDetector, EmotionAwareSarcasmDetector
from evaluation.evaluator import Evaluator


def load_config():
    """Load project configuration."""
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def get_emotion_predictions(emotion_clf, texts, cache_path, split_name):
    """Get emotion predictions, using cache if available."""
    cache_file = os.path.join(cache_path, f"emotion_cache_{split_name}.pkl")
    cached = load_emotion_cache(cache_file)
    if cached is not None and len(cached) == len(texts):
        return cached

    print(f"[Train] Computing emotion features for {split_name} ({len(texts)} samples)...")
    batch_size = 64
    predictions = emotion_clf.predict_batch(texts, batch_size=batch_size)
    cache_emotion_features(predictions, cache_file)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Train sarcasm detection models")
    parser.add_argument("--quick", action="store_true",
                        help="Use a subset of data for quick training")
    parser.add_argument("--no-transformer", action="store_true",
                        help="Use lexicon-based emotion classifier (no model download)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to use")
    args = parser.parse_args()

    print("=" * 70)
    print("EMOTION-AWARE SARCASM DETECTION SYSTEM - TRAINING PIPELINE")
    print("=" * 70)

    overall_start = time.time()
    config = load_config()

    # Directories
    data_dir = os.path.join(PROJECT_ROOT, "data")
    models_dir = os.path.join(PROJECT_ROOT, "saved_models")
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---- Step 1: Dataset ----
    print("\n[Step 1/6] Preparing dataset...")
    dm = DatasetManager(config, data_dir=data_dir)

    max_samples = args.max_samples
    if args.quick and max_samples is None:
        max_samples = 3000
        print("[Train] Quick mode: using 3000 samples")

    train_df, val_df, test_df = dm.split(max_samples=max_samples)
    dataset_info = dm.get_dataset_info()

    # ---- Step 2: Preprocessing ----
    print("\n[Step 2/6] Preprocessing text...")
    preprocessor = TextPreprocessor()

    train_texts = preprocessor.preprocess_batch(train_df["text"].tolist())
    val_texts = preprocessor.preprocess_batch(val_df["text"].tolist())
    test_texts = preprocessor.preprocess_batch(test_df["text"].tolist())

    train_labels = train_df["label"].values
    val_labels = val_df["label"].values
    test_labels = test_df["label"].values

    print(f"[Train] Preprocessed: {len(train_texts)} train, {len(val_texts)} val, {len(test_texts)} test")

    # ---- Step 3: Emotion features ----
    print("\n[Step 3/6] Computing emotion features...")
    use_transformer = not args.no_transformer
    emotion_clf = create_emotion_classifier(config, use_transformer=use_transformer)

    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    train_emotion_preds = get_emotion_predictions(emotion_clf, train_texts, cache_dir, "train")
    val_emotion_preds = get_emotion_predictions(emotion_clf, val_texts, cache_dir, "val")
    test_emotion_preds = get_emotion_predictions(emotion_clf, test_texts, cache_dir, "test")

    train_emotion_features = get_feature_vectors(train_emotion_preds)
    val_emotion_features = get_feature_vectors(val_emotion_preds)
    test_emotion_features = get_feature_vectors(test_emotion_preds)

    print(f"[Train] Emotion features shape: {train_emotion_features.shape}")

    # Show emotion distribution in training data
    train_emotion_labels = [p["label"] for p in train_emotion_preds]
    emotion_dist = pd.Series(train_emotion_labels).value_counts()
    print(f"[Train] Emotion distribution in training data:\n{emotion_dist.to_string()}")

    # ---- Step 4: Train baseline model ----
    print("\n[Step 4/6] Training baseline model...")
    baseline = BaselineSarcasmDetector(config)
    baseline.train(train_texts, train_labels)

    # Validate
    val_baseline_metrics = baseline.evaluate(val_texts, val_labels)
    print(f"[Baseline] Validation accuracy: {val_baseline_metrics['accuracy']:.4f}, "
          f"macro-F1: {val_baseline_metrics['macro_f1']:.4f}")

    # ---- Step 5: Train emotion-aware model ----
    print("\n[Step 5/6] Training emotion-aware model...")
    emotion_aware = EmotionAwareSarcasmDetector(config)
    emotion_aware.train(train_texts, train_labels, train_emotion_features)

    # Validate
    val_emotion_metrics = emotion_aware.evaluate(val_texts, val_labels, val_emotion_features)
    print(f"[EmotionAware] Validation accuracy: {val_emotion_metrics['accuracy']:.4f}, "
          f"macro-F1: {val_emotion_metrics['macro_f1']:.4f}")

    # ---- Step 6: Evaluate on held-out test set ----
    print("\n[Step 6/6] Evaluating on held-out test set...")
    print("[Train] The test set is used exactly once with frozen configuration (SRS constraint)")

    baseline_test_metrics = baseline.evaluate(test_texts, test_labels)
    emotion_test_metrics = emotion_aware.evaluate(test_texts, test_labels, test_emotion_features)

    # Compare models
    evaluator = Evaluator(results_dir)
    comparison = evaluator.compare_models(baseline_test_metrics, emotion_test_metrics)
    evaluator.print_comparison(comparison)

    # Save error samples
    baseline_preds, _ = baseline.predict_batch(test_texts)
    emotion_preds, _ = emotion_aware.predict_batch(test_texts, test_emotion_features)
    evaluator.save_error_samples(test_texts, test_labels, baseline_preds, "baseline")
    evaluator.save_error_samples(test_texts, test_labels, emotion_preds, "emotion_aware")

    # Save models
    baseline.save(models_dir)
    emotion_aware.save(models_dir)

    # Save config info
    import sklearn
    config_info = {
        "random_seed": config["random_seed"],
        "tfidf_max_features": config["tfidf_max_features"],
        "tfidf_ngram_range": config["tfidf_ngram_range"],
        "logistic_regression_C": config["logistic_regression_C"],
        "emotion_classifier_backend": emotion_clf.backend,
        "emotion_model": config.get("emotion_model_name", "lexicon"),
        "quick_mode": args.quick,
        "max_samples": max_samples,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
    }

    # Save full results
    evaluator.save_results(comparison, config_info, dataset_info)

    # ---- Summary ----
    total_time = round(time.time() - overall_start, 1)
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE in {total_time}s")
    print(f"Models saved to: {models_dir}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}")
    print(f"\nTo start the web application, run:")
    print(f"  python app.py")


if __name__ == "__main__":
    main()
