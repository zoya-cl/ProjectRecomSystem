import re
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from .base_engine import BaseEngine
from .advanced_preprocessor import AdvancedTextPreprocessor

class SmartTfidfEngine(BaseEngine):
    """
    An enhanced TF-IDF Engine with advanced domain-aware preprocessing.
    
    Uses AdvancedTextPreprocessor which:
    - Preserves domain-specific keywords (agriculture, blockchain, etc.)
    - Uses lemmatization instead of aggressive stemming
    - Detects technical phrases (machine learning, computer vision, etc.)
    - Protects programming languages and frameworks
    """
    
    def __init__(self, max_features: int = 15000, ngram_range: tuple = (1, 3)):
        """
        Initialize with enhanced features for better technical term matching.
        
        Args:
            max_features: Maximum number of features (increased to 15000 for better coverage)
            ngram_range: N-gram range for phrase detection (1-3 words)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # Let preprocessor handle stopwords
            lowercase=False,  # Preprocessor handles this
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w+\b',  # More flexible pattern
            sublinear_tf=True,  # Dampens the effect of very frequent terms
            min_df=1,  # Keep even rare technical terms
            dtype=np.float32
        )
        self.preprocessor = AdvancedTextPreprocessor()
        self._is_fitted = False

    def fit(self, documents: List[str]) -> None:
        processed = self.preprocessor.preprocess_list(documents)
        self.vectorizer.fit(processed)
        self._is_fitted = True

    def transform(self, documents: List[str]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Engine must be fitted before transform.")
        processed = self.preprocessor.preprocess_list(documents)
        return self.vectorizer.transform(processed).toarray()

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        processed = self.preprocessor.preprocess_list(documents)
        vectors = self.vectorizer.fit_transform(processed).toarray()
        self._is_fitted = True
        return vectors

    def get_similarity(self, query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        return cosine_similarity(query_vector, document_vectors).flatten()

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'preprocessor': self.preprocessor,  # Save preprocessor too!
                'is_fitted': self._is_fitted
            }, f)

    @classmethod
    def load(cls, path: str) -> "SmartTfidfEngine":
        with open(path, 'rb') as f:
            data = pickle.load(f)
        engine = cls()
        engine.vectorizer = data['vectorizer']
        engine.preprocessor = data.get('preprocessor', AdvancedTextPreprocessor())  # Restore or create new
        engine._is_fitted = data['is_fitted']
        return engine

