"""
Emotion classifier for the Emotion-Aware Sarcasm Detection System.

Provides emotion analysis as an auxiliary signal for sarcasm detection (FR-3.1 to FR-3.4).
Uses a pretrained transformer model with a lexicon-based fallback.
The output is an estimated emotion category - not a verified emotional state (FR-3.4).
"""

import os
import pickle
import numpy as np
from tqdm import tqdm

# Standard emotion labels used across both classifier backends
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class TransformerEmotionClassifier:
    """Emotion classifier using a pretrained distilroberta model.

    Model: j-hartmann/emotion-english-distilroberta-base
    Labels: anger, disgust, fear, joy, neutral, sadness, surprise
    """

    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.pipeline = None
        self.is_loaded = False
        self.backend = "transformer"

    def load(self):
        """Load the transformer model. Called lazily on first use."""
        if self.is_loaded:
            return
        try:
            from transformers import pipeline as hf_pipeline
            print(f"[Emotion] Loading transformer model: {self.model_name}")
            self.pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                truncation=True,
                max_length=128,
            )
            self.is_loaded = True
            print("[Emotion] Transformer model loaded successfully")
        except Exception as e:
            print(f"[Emotion] Failed to load transformer model: {e}")
            raise

    def predict(self, text):
        """Predict emotion for a single text.

        Args:
            text: Input text string.

        Returns:
            dict with 'label' (str) and 'scores' (dict of emotion -> probability).
        """
        if not self.is_loaded:
            self.load()

        if not text or not text.strip():
            return self._neutral_prediction()

        try:
            results = self.pipeline(text[:512])[0]
            scores = {r["label"]: round(float(r["score"]), 4) for r in results}
            # Ensure all labels are present
            for label in EMOTION_LABELS:
                if label not in scores:
                    scores[label] = 0.0
            label = max(scores, key=scores.get)
            return {"label": label, "scores": scores}
        except Exception:
            return self._neutral_prediction()

    def predict_batch(self, texts, batch_size=32):
        """Predict emotions for a batch of texts with progress bar.

        Args:
            texts: List of input text strings.
            batch_size: Number of texts to process at once.

        Returns:
            List of prediction dicts.
        """
        if not self.is_loaded:
            self.load()

        all_results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Emotion inference"):
            batch = [t[:512] if t else "" for t in texts[i : i + batch_size]]
            try:
                results = self.pipeline(batch, batch_size=batch_size)
                for result in results:
                    scores = {r["label"]: round(float(r["score"]), 4) for r in result}
                    for label in EMOTION_LABELS:
                        if label not in scores:
                            scores[label] = 0.0
                    label = max(scores, key=scores.get)
                    all_results.append({"label": label, "scores": scores})
            except Exception:
                # Fallback for failed batches
                for _ in batch:
                    all_results.append(self._neutral_prediction())

        return all_results

    def _neutral_prediction(self):
        scores = {label: 0.0 for label in EMOTION_LABELS}
        scores["neutral"] = 1.0
        return {"label": "neutral", "scores": scores}


class LexiconEmotionClassifier:
    """Fallback emotion classifier using keyword/lexicon matching.

    Used when transformers/torch is not available or model download fails.
    Less accurate but works without any downloads.
    """

    # Curated emotion word lists
    EMOTION_LEXICON = {
        "joy": [
            "happy", "glad", "wonderful", "great", "love", "excellent", "amazing",
            "fantastic", "beautiful", "brilliant", "cheerful", "delighted", "enjoy",
            "excited", "fun", "good", "laugh", "nice", "perfect", "pleased",
            "smile", "terrific", "thrilled", "win", "won", "best", "celebrate",
            "congratulations", "proud", "success", "awesome", "magnificent",
        ],
        "anger": [
            "angry", "furious", "outraged", "hate", "rage", "mad", "annoyed",
            "irritated", "frustrated", "hostile", "bitter", "disgusted", "enraged",
            "infuriated", "livid", "offended", "provoked", "resentful", "upset",
            "vicious", "violent", "wrath", "damn", "stupid", "idiot", "worst",
        ],
        "sadness": [
            "sad", "depressed", "miserable", "grief", "cry", "tears", "sorrow",
            "heartbroken", "lonely", "melancholy", "mourn", "pain", "suffer",
            "tragic", "unhappy", "disappointed", "hopeless", "regret", "sorry",
            "devastating", "gloomy", "lost", "miss", "poor", "unfortunately",
        ],
        "fear": [
            "afraid", "scared", "terrified", "worried", "anxious", "danger",
            "dread", "frightened", "horror", "panic", "nervous", "alarm",
            "concern", "creepy", "eerie", "nightmare", "shock", "startled",
            "suspicious", "tense", "threat", "uneasy", "warn", "risk",
        ],
        "surprise": [
            "surprised", "shocked", "amazing", "unexpected", "astonished",
            "astounding", "incredible", "remarkable", "stunning", "unbelievable",
            "whoa", "wow", "omg", "suddenly", "breaking", "revelation",
            "bizarre", "extraordinary", "jaw-dropping", "miracle", "twist",
        ],
        "disgust": [
            "disgusting", "gross", "revolting", "awful", "horrible", "nasty",
            "repulsive", "sickening", "terrible", "vile", "yuck", "appalling",
            "despicable", "dreadful", "foul", "loathsome", "nauseating",
            "obnoxious", "offensive", "repugnant", "shameful", "ugly", "vomit",
        ],
    }

    def __init__(self):
        self.backend = "lexicon"
        self.is_loaded = True

    def load(self):
        """No-op for the lexicon classifier (always ready)."""
        pass

    def predict(self, text):
        """Predict emotion using keyword matching.

        Args:
            text: Input text string.

        Returns:
            dict with 'label' (str) and 'scores' (dict of emotion -> score).
        """
        if not text or not text.strip():
            return self._neutral_prediction()

        words = text.lower().split()
        scores = {}

        for emotion, keywords in self.EMOTION_LEXICON.items():
            count = sum(1 for w in words if w.strip(".,!?;:'\"") in keywords)
            scores[emotion] = count

        # Add neutral
        total = sum(scores.values())
        if total == 0:
            return self._neutral_prediction()

        # Normalise to probabilities
        scores["neutral"] = max(0, len(words) * 0.1 - total)
        total = sum(scores.values())
        scores = {k: round(v / total, 4) for k, v in scores.items()}

        label = max(scores, key=scores.get)
        return {"label": label, "scores": scores}

    def predict_batch(self, texts, batch_size=32):
        """Predict emotions for a batch of texts.

        Args:
            texts: List of input text strings.
            batch_size: Ignored (instant processing).

        Returns:
            List of prediction dicts.
        """
        return [self.predict(t) for t in tqdm(texts, desc="Emotion inference (lexicon)")]

    def _neutral_prediction(self):
        scores = {label: 0.0 for label in EMOTION_LABELS}
        scores["neutral"] = 1.0
        return {"label": "neutral", "scores": scores}


def create_emotion_classifier(config, use_transformer=True):
    """Factory function to create the best available emotion classifier.

    Tries the transformer model first; falls back to lexicon-based if unavailable.

    Args:
        config: Project configuration dict.
        use_transformer: Whether to attempt loading the transformer model.

    Returns:
        An emotion classifier instance (Transformer or Lexicon).
    """
    if use_transformer:
        try:
            clf = TransformerEmotionClassifier(config.get("emotion_model_name",
                "j-hartmann/emotion-english-distilroberta-base"))
            clf.load()
            return clf
        except Exception as e:
            print(f"[Emotion] Transformer unavailable ({e}). Using lexicon fallback.")

    print("[Emotion] Using lexicon-based emotion classifier")
    return LexiconEmotionClassifier()


def get_feature_vector(prediction):
    """Convert an emotion prediction to a fixed-size feature vector.

    Args:
        prediction: dict with 'scores' mapping emotion labels to probabilities.

    Returns:
        List of floats in EMOTION_LABELS order.
    """
    return [prediction["scores"].get(label, 0.0) for label in EMOTION_LABELS]


def get_feature_vectors(predictions):
    """Convert a list of emotion predictions to a feature matrix.

    Args:
        predictions: List of prediction dicts.

    Returns:
        numpy array of shape (n_samples, n_emotions).
    """
    return np.array([get_feature_vector(p) for p in predictions])


def cache_emotion_features(predictions, cache_path):
    """Save computed emotion predictions to disk for reuse."""
    with open(cache_path, "wb") as f:
        pickle.dump(predictions, f)
    print(f"[Emotion] Cached {len(predictions)} predictions to {cache_path}")


def load_emotion_cache(cache_path):
    """Load cached emotion predictions from disk."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            predictions = pickle.load(f)
        print(f"[Emotion] Loaded {len(predictions)} cached predictions from {cache_path}")
        return predictions
    return None
