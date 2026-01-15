"""
TF-IDF Vectorization Engine.

This is the default engine using scikit-learn's TfidfVectorizer.
To switch to a different engine, create a new engine file and update build_pipeline.py.
"""

import pickle
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base_engine import BaseEngine


class TfidfEngine(BaseEngine):
    """
    TF-IDF based vectorization engine using scikit-learn.
    
    Features:
    - Fast and lightweight
    - No external API calls
    - Good for keyword-based matching
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Initialize TF-IDF engine.
        
        Args:
            max_features: Maximum vocabulary size.
            ngram_range: Range of n-grams to extract.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            dtype=np.float32
        )
        self._is_fitted = False
    
    def fit(self, documents: List[str]) -> None:
        """Fit the TF-IDF vectorizer on documents."""
        self.vectorizer.fit(documents)
        self._is_fitted = True
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF vectors."""
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform. Call fit() first.")
        return self.vectorizer.transform(documents).toarray()
    
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        vectors = self.vectorizer.fit_transform(documents).toarray()
        self._is_fitted = True
        return vectors
    
    def get_similarity(self, query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents."""
        # Ensure query_vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        return cosine_similarity(query_vector, document_vectors).flatten()
    
    def save(self, path: str) -> None:
        """Save the fitted vectorizer to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'is_fitted': self._is_fitted
            }, f)
        print(f"[TfidfEngine] Saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "TfidfEngine":
        """Load a previously saved TF-IDF engine."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        engine = cls()
        engine.vectorizer = data['vectorizer']
        engine._is_fitted = data['is_fitted']
        print(f"[TfidfEngine] Loaded from {path}")
        return engine
