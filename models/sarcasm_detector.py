"""
Sarcasm detection models for the Emotion-Aware Sarcasm Detection System.

Implements:
1. BaselineSarcasmDetector: TF-IDF + Logistic Regression (text-only)
2. EmotionAwareSarcasmDetector: TF-IDF + emotion features + Logistic Regression (late fusion)

Both models use the same data splits and evaluation measures (FR-5.2).
Vectorisers are fitted on training data only (Design constraint in SRS).
"""

import os
import time
import numpy as np
import joblib
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class BaselineSarcasmDetector:
    """Text-only sarcasm classifier using TF-IDF features and Logistic Regression.

    This serves as the baseline against which the emotion-aware model is compared.
    """

    def __init__(self, config):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=config.get("tfidf_max_features", 10000),
            ngram_range=tuple(config.get("tfidf_ngram_range", [1, 2])),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.classifier = LogisticRegression(
            C=config.get("logistic_regression_C", 1.0),
            max_iter=config.get("logistic_regression_max_iter", 1000),
            random_state=config.get("random_seed", 42),
            solver="lbfgs",
            class_weight="balanced",
        )
        self.is_trained = False
        self.model_name = "baseline"
        self.training_time = None

    def train(self, texts, labels):
        """Train the baseline model.

        Vectoriser is fitted on training data only (SRS design constraint).

        Args:
            texts: List of preprocessed text strings.
            labels: List/array of binary labels (0=non-sarcastic, 1=sarcastic).
        """
        print("[Baseline] Training TF-IDF + Logistic Regression ...")
        start = time.time()

        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)

        self.training_time = round(time.time() - start, 2)
        self.is_trained = True
        print(f"[Baseline] Training complete in {self.training_time}s")
        print(f"[Baseline] TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def predict(self, text):
        """Predict sarcasm for a single text.

        Args:
            text: Preprocessed text string.

        Returns:
            dict with 'label' (str), 'is_sarcastic' (int) and 'confidence' (float).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run train() first.")

        X = self.vectorizer.transform([text])
        prob = self.classifier.predict_proba(X)[0]
        pred = int(self.classifier.predict(X)[0])
        confidence = float(prob[pred])

        return {
            "label": "sarcastic" if pred == 1 else "non-sarcastic",
            "is_sarcastic": pred,
            "confidence": round(confidence, 4),
            "probabilities": {
                "non-sarcastic": round(float(prob[0]), 4),
                "sarcastic": round(float(prob[1]), 4),
            },
        }

    def predict_batch(self, texts):
        """Predict sarcasm for a batch of texts.

        Args:
            texts: List of preprocessed text strings.

        Returns:
            Tuple of (predictions, probabilities) as numpy arrays.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run train() first.")

        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        return predictions, probabilities

    def evaluate(self, texts, labels):
        """Evaluate the model on a dataset.

        Args:
            texts: List of preprocessed text strings.
            labels: List/array of true labels.

        Returns:
            dict with evaluation metrics.
        """
        start = time.time()
        predictions, probabilities = self.predict_batch(texts)
        latency_total = time.time() - start
        latency_per_sample = latency_total / len(texts) if texts else 0

        metrics = {
            "accuracy": round(float(accuracy_score(labels, predictions)), 4),
            "precision": round(float(precision_score(labels, predictions, average="binary")), 4),
            "recall": round(float(recall_score(labels, predictions, average="binary")), 4),
            "f1_score": round(float(f1_score(labels, predictions, average="binary")), 4),
            "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 4),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
            "classification_report": classification_report(labels, predictions, 
                target_names=["non-sarcastic", "sarcastic"], output_dict=True),
            "total_samples": len(labels),
            "total_latency_seconds": round(latency_total, 4),
            "latency_per_sample_ms": round(latency_per_sample * 1000, 2),
        }

        return metrics

    def save(self, directory):
        """Save model artifacts to disk."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(directory, "baseline_vectorizer.pkl"))
        joblib.dump(self.classifier, os.path.join(directory, "baseline_classifier.pkl"))
        print(f"[Baseline] Model saved to {directory}")

    def load(self, directory):
        """Load model artifacts from disk."""
        self.vectorizer = joblib.load(os.path.join(directory, "baseline_vectorizer.pkl"))
        self.classifier = joblib.load(os.path.join(directory, "baseline_classifier.pkl"))
        self.is_trained = True
        print(f"[Baseline] Model loaded from {directory}")


class EmotionAwareSarcasmDetector:
    """Emotion-aware sarcasm classifier using TF-IDF + emotion features (late fusion).

    Combines text features (TF-IDF) with emotion probability vectors from the
    emotion classifier. The emotion features are concatenated with text features
    before the final classification (late fusion approach, as described in SRS).

    At prediction time, emotion features are generated only from the submitted text;
    ground-truth emotion labels are never supplied (FR-4.4).
    """

    def __init__(self, config):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=config.get("tfidf_max_features", 10000),
            ngram_range=tuple(config.get("tfidf_ngram_range", [1, 2])),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.classifier = LogisticRegression(
            C=config.get("logistic_regression_C", 1.0),
            max_iter=config.get("logistic_regression_max_iter", 1000),
            random_state=config.get("random_seed", 42),
            solver="lbfgs",
            class_weight="balanced",
        )
        self.is_trained = False
        self.model_name = "emotion_aware"
        self.training_time = None

    def _combine_features(self, tfidf_matrix, emotion_features):
        """Combine TF-IDF and emotion features via late fusion (concatenation).

        Args:
            tfidf_matrix: Sparse TF-IDF feature matrix.
            emotion_features: Dense emotion probability matrix (n_samples x n_emotions).

        Returns:
            Combined feature matrix.
        """
        from scipy.sparse import csr_matrix
        emotion_sparse = csr_matrix(emotion_features)
        return hstack([tfidf_matrix, emotion_sparse])

    def train(self, texts, labels, emotion_features):
        """Train the emotion-aware model.

        Args:
            texts: List of preprocessed text strings.
            labels: List/array of binary labels.
            emotion_features: numpy array of emotion feature vectors (n_samples x n_emotions).
        """
        print("[EmotionAware] Training TF-IDF + Emotion + Logistic Regression ...")
        start = time.time()

        X_tfidf = self.vectorizer.fit_transform(texts)
        X_combined = self._combine_features(X_tfidf, emotion_features)
        self.classifier.fit(X_combined, labels)

        self.training_time = round(time.time() - start, 2)
        self.is_trained = True
        print(f"[EmotionAware] Training complete in {self.training_time}s")
        print(f"[EmotionAware] Feature dimensions: TF-IDF={X_tfidf.shape[1]}, "
              f"Emotion={emotion_features.shape[1]}, "
              f"Combined={X_combined.shape[1]}")

    def predict(self, text, emotion_feature_vector):
        """Predict sarcasm for a single text with emotion features.

        Args:
            text: Preprocessed text string.
            emotion_feature_vector: List/array of emotion probabilities.

        Returns:
            dict with 'label', 'is_sarcastic' and 'confidence'.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run train() first.")

        X_tfidf = self.vectorizer.transform([text])
        emotion_array = np.array(emotion_feature_vector).reshape(1, -1)
        X_combined = self._combine_features(X_tfidf, emotion_array)

        prob = self.classifier.predict_proba(X_combined)[0]
        pred = int(self.classifier.predict(X_combined)[0])
        confidence = float(prob[pred])

        return {
            "label": "sarcastic" if pred == 1 else "non-sarcastic",
            "is_sarcastic": pred,
            "confidence": round(confidence, 4),
            "probabilities": {
                "non-sarcastic": round(float(prob[0]), 4),
                "sarcastic": round(float(prob[1]), 4),
            },
        }

    def predict_batch(self, texts, emotion_features):
        """Predict sarcasm for a batch of texts.

        Args:
            texts: List of preprocessed text strings.
            emotion_features: numpy array of emotion feature vectors.

        Returns:
            Tuple of (predictions, probabilities).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run train() first.")

        X_tfidf = self.vectorizer.transform(texts)
        X_combined = self._combine_features(X_tfidf, emotion_features)
        predictions = self.classifier.predict(X_combined)
        probabilities = self.classifier.predict_proba(X_combined)
        return predictions, probabilities

    def evaluate(self, texts, labels, emotion_features):
        """Evaluate the model on a dataset.

        Args:
            texts: List of preprocessed text strings.
            labels: List/array of true labels.
            emotion_features: numpy array of emotion feature vectors.

        Returns:
            dict with evaluation metrics.
        """
        start = time.time()
        predictions, probabilities = self.predict_batch(texts, emotion_features)
        latency_total = time.time() - start
        latency_per_sample = latency_total / len(texts) if texts else 0

        metrics = {
            "accuracy": round(float(accuracy_score(labels, predictions)), 4),
            "precision": round(float(precision_score(labels, predictions, average="binary")), 4),
            "recall": round(float(recall_score(labels, predictions, average="binary")), 4),
            "f1_score": round(float(f1_score(labels, predictions, average="binary")), 4),
            "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 4),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
            "classification_report": classification_report(labels, predictions,
                target_names=["non-sarcastic", "sarcastic"], output_dict=True),
            "total_samples": len(labels),
            "total_latency_seconds": round(latency_total, 4),
            "latency_per_sample_ms": round(latency_per_sample * 1000, 2),
            "emotion_feature_dim": emotion_features.shape[1],
        }

        return metrics

    def save(self, directory):
        """Save model artifacts to disk."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(directory, "emotion_aware_vectorizer.pkl"))
        joblib.dump(self.classifier, os.path.join(directory, "emotion_aware_classifier.pkl"))
        print(f"[EmotionAware] Model saved to {directory}")

    def load(self, directory):
        """Load model artifacts from disk."""
        self.vectorizer = joblib.load(os.path.join(directory, "emotion_aware_vectorizer.pkl"))
        self.classifier = joblib.load(os.path.join(directory, "emotion_aware_classifier.pkl"))
        self.is_trained = True
        print(f"[EmotionAware] Model loaded from {directory}")
