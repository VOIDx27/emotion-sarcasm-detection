"""
Evaluation pipeline for the Emotion-Aware Sarcasm Detection System.

Compares the text-only baseline with the emotion-aware model on the same
held-out test set using agreed metrics (FR-5.1 to FR-5.6).

Metrics: accuracy, precision, recall, macro-F1, confusion matrix, latency.
"""

import json
import os
import time
import pandas as pd
import numpy as np


class Evaluator:
    """Runs evaluation and comparison of sarcasm detection models."""

    def __init__(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def compare_models(self, baseline_metrics, emotion_aware_metrics):
        """Compare baseline and emotion-aware model metrics.

        Args:
            baseline_metrics: dict of metrics from baseline model.
            emotion_aware_metrics: dict of metrics from emotion-aware model.

        Returns:
            dict with comparison results and improvement analysis.
        """
        comparison = {
            "baseline": baseline_metrics,
            "emotion_aware": emotion_aware_metrics,
            "improvement": {},
        }

        # Calculate improvements
        for metric in ["accuracy", "precision", "recall", "f1_score", "macro_f1"]:
            baseline_val = baseline_metrics.get(metric, 0)
            emotion_val = emotion_aware_metrics.get(metric, 0)
            diff = emotion_val - baseline_val
            comparison["improvement"][metric] = {
                "absolute": round(diff, 4),
                "relative_percent": round((diff / baseline_val * 100) if baseline_val > 0 else 0, 2),
                "improved": diff > 0,
            }

        # Overall verdict
        macro_f1_diff = comparison["improvement"]["macro_f1"]["absolute"]
        if macro_f1_diff > 0.005:
            comparison["verdict"] = "Emotion features IMPROVED sarcasm detection"
        elif macro_f1_diff < -0.005:
            comparison["verdict"] = "Emotion features DID NOT improve sarcasm detection"
        else:
            comparison["verdict"] = "Emotion features had NEGLIGIBLE effect on sarcasm detection"

        comparison["note"] = (
            "This comparison is based on the same held-out test split. "
            "Results hold only for the evaluated data and should not be overgeneralised."
        )

        return comparison

    def save_results(self, comparison, config_info, dataset_info):
        """Save evaluation results to JSON files.

        Args:
            comparison: dict from compare_models().
            config_info: dict of configuration used for this run.
            dataset_info: dict of dataset metadata.
        """
        # Full results
        full_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "comparison": comparison,
            "configuration": config_info,
            "dataset": dataset_info,
        }

        results_path = os.path.join(self.results_dir, "evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, default=str)

        print(f"[Evaluation] Results saved to {results_path}")
        return results_path

    def save_error_samples(self, texts, true_labels, pred_labels, model_name, n_samples=50):
        """Save a sample of misclassified examples for error analysis (FR-5.6).

        Args:
            texts: List of text strings.
            true_labels: Array of true labels.
            pred_labels: Array of predicted labels.
            model_name: Name of the model ('baseline' or 'emotion_aware').
            n_samples: Maximum number of error samples to save.

        Returns:
            DataFrame of error samples.
        """
        errors = []
        for i, (text, true, pred) in enumerate(zip(texts, true_labels, pred_labels)):
            if true != pred:
                error_type = "false_positive" if pred == 1 else "false_negative"
                errors.append({
                    "index": i,
                    "text": text,
                    "true_label": "sarcastic" if true == 1 else "non-sarcastic",
                    "predicted_label": "sarcastic" if pred == 1 else "non-sarcastic",
                    "error_type": error_type,
                })

        df_errors = pd.DataFrame(errors)

        if len(df_errors) > n_samples:
            # Sample equally from false positives and false negatives
            fp = df_errors[df_errors["error_type"] == "false_positive"]
            fn = df_errors[df_errors["error_type"] == "false_negative"]
            n_each = n_samples // 2
            fp_sample = fp.head(n_each)
            fn_sample = fn.head(n_each)
            df_errors = pd.concat([fp_sample, fn_sample]).reset_index(drop=True)

        error_path = os.path.join(self.results_dir, f"error_samples_{model_name}.csv")
        df_errors.to_csv(error_path, index=False)
        print(f"[Evaluation] {len(df_errors)} error samples saved to {error_path}")

        return df_errors

    def print_comparison(self, comparison):
        """Pretty-print the model comparison to console."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS - MODEL COMPARISON")
        print("=" * 70)

        for model_name in ["baseline", "emotion_aware"]:
            metrics = comparison[model_name]
            display_name = "Text-Only Baseline" if model_name == "baseline" else "Emotion-Aware Model"
            print(f"\n--- {display_name} ---")
            print(f"  Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  Precision:    {metrics['precision']:.4f}")
            print(f"  Recall:       {metrics['recall']:.4f}")
            print(f"  F1-Score:     {metrics['f1_score']:.4f}")
            print(f"  Macro-F1:     {metrics['macro_f1']:.4f}")
            print(f"  Latency/sample: {metrics['latency_per_sample_ms']:.2f} ms")
            cm = metrics["confusion_matrix"]
            print(f"  Confusion Matrix:")
            print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
            print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

        print(f"\n--- Improvement (Emotion-Aware vs Baseline) ---")
        for metric, info in comparison["improvement"].items():
            if info["improved"]:
                direction = "[UP]"
            elif info["absolute"] < 0:
                direction = "[DOWN]"
            else:
                direction = "[SAME]"
            print(f"  {metric:12s}: {info['absolute']:+.4f} ({info['relative_percent']:+.2f}%) {direction}")

        print(f"\n  Verdict: {comparison['verdict']}")
        print(f"  Note: {comparison['note']}")
        print("=" * 70 + "\n")
