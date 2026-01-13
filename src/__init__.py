"""
LinkedIn Smart Recommender
==========================

A smart recommendation system for LinkedIn using NLP and Machine Learning.

Modules:
    - config: Configuration management
    - data_loader: Data loading and preprocessing
    - embeddings: NLP embedding generation
    - scoring: Multi-criteria scoring system
    - recommender: Main recommendation engine
    - utils: Utility functions
"""

from .config import config, AppConfig
from .data_loader import DataLoader
from .embeddings import EmbeddingEngine
from .scoring import ScoringEngine
from .recommender import LinkedInRecommender
from .utils import normalize_text, calculate_similarity

__version__ = "1.0.0"
__author__ = "LinkedIn Recommender Team"

__all__ = [
    "config",
    "AppConfig",
    "DataLoader",
    "EmbeddingEngine", 
    "ScoringEngine",
    "LinkedInRecommender",
    "normalize_text",
    "calculate_similarity"
]
