#!/usr/bin/env python3
"""
Recommender Module - Load artifacts and run queries.

This module provides the ProjectRecommender class that:
- Loads pre-computed artifacts (no recomputation)
- Provides recommend() method for text-based search
- Supports filtering by domain, language, difficulty

Usage:
    from recommender import ProjectRecommender
    
    rec = ProjectRecommender()
    results = rec.recommend("machine learning python", top_n=5)
    for r in results:
        print(r['title'], r['score'])
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# Import engine class for loading
from engines.smart_tfidf_engine import SmartTfidfEngine

# Artifact paths
ARTIFACTS_DIR = Path("artifacts")
ENGINE_PATH = ARTIFACTS_DIR / "engine.pkl"
VECTORS_PATH = ARTIFACTS_DIR / "vectors.npy"
DATAFRAME_PATH = ARTIFACTS_DIR / "processed_data.pkl"


class ProjectRecommender:
    """
    Project Recommendation Engine.
    
    Loads pre-computed artifacts and provides fast similarity search.
    """
    
    def __init__(self, artifacts_dir: Optional[Path] = None):
        """
        Initialize recommender by loading artifacts.
        
        Args:
            artifacts_dir: Optional custom artifacts directory.
        """
        if artifacts_dir:
            self.artifacts_dir = Path(artifacts_dir)
        else:
            self.artifacts_dir = ARTIFACTS_DIR
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all pre-computed artifacts."""
        engine_path = self.artifacts_dir / "engine.pkl"
        vectors_path = self.artifacts_dir / "vectors.npy"
        dataframe_path = self.artifacts_dir / "processed_data.pkl"
        
        # Check if artifacts exist
        for path in [engine_path, vectors_path, dataframe_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Artifact not found: {path}\n"
                    "Run 'python build_pipeline.py' first to build artifacts."
                )
        
        # Load engine
        self.engine = SmartTfidfEngine.load(str(engine_path))
        
        # Load vectors
        self.vectors = np.load(str(vectors_path))
        print(f"[Recommender] Loaded vectors: {self.vectors.shape}")
        
        # Load dataframe
        with open(dataframe_path, 'rb') as f:
            self.df = pickle.load(f)
        print(f"[Recommender] Loaded {len(self.df)} projects")
    
    def get_domains(self) -> List[str]:
        """Get list of unique domains."""
        domains = self.df['Domain'].dropna().unique().tolist()
        return sorted(set(d.strip() for d in domains if d.strip()))
    
    def get_languages(self) -> List[str]:
        """Get list of unique programming languages."""
        languages = self.df['Language Used'].dropna().unique().tolist()
        return sorted(set(l.strip() for l in languages if l.strip()))
    
    def get_difficulty_levels(self) -> List[str]:
        """Get list of difficulty levels."""
        levels = self.df['Toughness Level'].dropna().unique().tolist()
        return sorted(set(l.strip() for l in levels if l.strip()))
    
    def recommend(
        self,
        query: str,
        top_n: int = 10,
        domain_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        difficulty_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get project recommendations based on query.
        
        Args:
            query: Search query (keywords, description, etc.)
            top_n: Number of results to return.
            domain_filter: Filter by domain (case-insensitive partial match).
            language_filter: Filter by programming language (case-insensitive partial match).
            difficulty_filter: Filter by difficulty level (case-insensitive partial match).
            min_score: Minimum similarity score threshold.
            
        Returns:
            List of recommendation dictionaries with project details and scores.
        """
        # Transform query to vector
        query_vector = self.engine.transform([query])
        
        # Compute similarities
        similarities = self.engine.get_similarity(query_vector, self.vectors)
        
        # Create results dataframe
        results_df = self.df.copy()
        results_df['similarity_score'] = similarities
        
        # Apply filters
        if domain_filter:
            domain_lower = domain_filter.lower()
            results_df = results_df[
                results_df['domain_clean'].str.contains(domain_lower, na=False)
            ]
        
        if language_filter:
            lang_lower = language_filter.lower()
            results_df = results_df[
                results_df['language_clean'].str.contains(lang_lower, na=False)
            ]
        
        if difficulty_filter:
            diff_lower = difficulty_filter.lower()
            results_df = results_df[
                results_df['toughness_clean'].str.contains(diff_lower, na=False)
            ]
        
        # Filter by minimum score
        if min_score > 0:
            results_df = results_df[results_df['similarity_score'] >= min_score]
        
        # Sort by similarity and get top N
        results_df = results_df.sort_values('similarity_score', ascending=False).head(top_n)
        
        # Format results
        results = []
        for _, row in results_df.iterrows():
            results.append({
                'project_id': row.get('Project ID', ''),
                'title': row.get('Title of the Project', ''),
                'domain': row.get('Domain', ''),
                'description': row.get('Short Description', ''),
                'language': row.get('Language Used', ''),
                'difficulty': row.get('Toughness Level', ''),
                'keywords': row.get('Keywords', ''),
                'libraries': row.get('Toolkits/Libraries', ''),
                'build_time': row.get('Expected Build Time', ''),
                'documentation': row.get('Documentation', ''),
                'score': round(float(row['similarity_score']), 4)
            })
        
        return results
    
    def recommend_by_project_id(self, project_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar projects to a given project ID.
        
        Args:
            project_id: The project ID to find similar projects for.
            top_n: Number of similar projects to return.
            
        Returns:
            List of similar project recommendations.
        """
        # Find the project
        project_mask = self.df['Project ID'] == project_id
        if not project_mask.any():
            raise ValueError(f"Project ID {project_id} not found.")
        
        project_idx = self.df[project_mask].index[0]
        project_vector = self.vectors[project_idx].reshape(1, -1)
        
        # Get similarities
        similarities = self.engine.get_similarity(project_vector, self.vectors)
        
        # Create results (excluding the query project itself)
        results_df = self.df.copy()
        results_df['similarity_score'] = similarities
        results_df = results_df[results_df['Project ID'] != project_id]
        results_df = results_df.sort_values('similarity_score', ascending=False).head(top_n)
        
        results = []
        for _, row in results_df.iterrows():
            results.append({
                'project_id': row.get('Project ID', ''),
                'title': row.get('Title of the Project', ''),
                'domain': row.get('Domain', ''),
                'description': row.get('Short Description', ''),
                'score': round(float(row['similarity_score']), 4)
            })
        
        return results


# Quick test
if __name__ == "__main__":
    rec = ProjectRecommender()
    
    print("\n" + "="*60)
    print("SAMPLE RECOMMENDATION")
    print("="*60)
    
    results = rec.recommend("machine learning python classification", top_n=5)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['title']} (Score: {r['score']})")
        print(f"   Domain: {r['domain']}")
        print(f"   Language: {r['language']}")
