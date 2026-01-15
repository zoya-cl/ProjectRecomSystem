"""
Base Engine Interface for Recommendation System.

To create a new vectorization engine:
1. Create a new file in engines/ (e.g., sentence_transformer_engine.py)
2. Subclass BaseEngine
3. Implement fit(), transform(), get_similarity(), save(), load()
4. Update build_pipeline.py to use your new engine
"""

from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np


class BaseEngine(ABC):
    """
    Abstract base class for vectorization engines.
    
    Subclass this to implement different vectorization strategies:
    - TF-IDF (current default)
    - Sentence Transformers (BERT-based)
    - Word2Vec / FastText
    - OpenAI Embeddings
    - etc.
    """
    
    @abstractmethod
    def fit(self, documents: List[str]) -> None:
        """
        Fit the vectorizer on a list of documents.
        
        Args:
            documents: List of text documents to fit on.
        """
        pass
    
    @abstractmethod
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents into vectors.
        
        Args:
            documents: List of text documents to transform.
            
        Returns:
            np.ndarray: Matrix of document vectors.
        """
        pass
    
    @abstractmethod
    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            documents: List of text documents.
            
        Returns:
            np.ndarray: Matrix of document vectors.
        """
        pass
    
    @abstractmethod
    def get_similarity(self, query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
        """
        Compute similarity between query and all documents.
        
        Args:
            query_vector: Vector representation of query.
            document_vectors: Matrix of document vectors.
            
        Returns:
            np.ndarray: Similarity scores for each document.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the fitted engine to disk.
        
        Args:
            path: Path to save the engine.
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseEngine":
        """
        Load a previously saved engine from disk.
        
        Args:
            path: Path to the saved engine.
            
        Returns:
            BaseEngine: Loaded engine instance.
        """
        pass
