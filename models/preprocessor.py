"""
Text preprocessor for the Emotion-Aware Sarcasm Detection System.

Handles tokenisation and normalisation (FR-2.1 to FR-2.4).
Preserves punctuation, casing and expressive tokens that can signal sarcasm (FR-2.3).
Consistent between training and inference (FR-2.4).
"""

import re


class TextPreprocessor:
    """Lightweight text preprocessor designed for sarcasm detection.

    Key design decisions:
    - Preserves punctuation (! ? ... are strong sarcasm cues)
    - Preserves casing (ALL CAPS often signals sarcasm)
    - Preserves expressive lengthening (e.g., 'sooooo')
    - Removes URLs and HTML tags (noise, not sarcasm cues)
    - Normalises whitespace only
    """

    def __init__(self):
        # Compiled regex patterns for efficiency
        self._url_pattern = re.compile(r"http\S+|www\.\S+")
        self._html_pattern = re.compile(r"<[^>]+>")
        self._whitespace_pattern = re.compile(r"\s+")
        self._special_chars_pattern = re.compile(r"[^\w\s!?.,;:'\"-]")

    def preprocess(self, text):
        """Preprocess a single text string.

        Args:
            text: Input text string.

        Returns:
            Cleaned text string, or empty string if input is invalid.
        """
        if not text or not isinstance(text, str):
            return ""

        # Remove URLs
        text = self._url_pattern.sub("", text)

        # Remove HTML tags
        text = self._html_pattern.sub("", text)

        # Remove unusual special characters but keep punctuation
        text = self._special_chars_pattern.sub(" ", text)

        # Normalise whitespace
        text = self._whitespace_pattern.sub(" ", text).strip()

        return text

    def preprocess_batch(self, texts):
        """Preprocess a list of text strings.

        Args:
            texts: List of input text strings.

        Returns:
            List of cleaned text strings.
        """
        return [self.preprocess(t) for t in texts]

    def validate_input(self, text, min_length=3, max_length=500):
        """Validate user input (FR-1.3, FR-1.4).

        Args:
            text: Raw input text.
            min_length: Minimum allowed length.
            max_length: Maximum allowed length.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not text or not isinstance(text, str):
            return False, "Please enter some text to analyse."

        text = text.strip()

        if len(text) < min_length:
            return False, f"Text is too short. Please enter at least {min_length} characters."

        if len(text) > max_length:
            return False, f"Text is too long. Maximum length is {max_length} characters."

        return True, None
