"""
Embedding Engine Module
=======================

Generates semantic embeddings using Sentence Transformers for text similarity.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import hashlib
import pickle
from datetime import datetime, timedelta
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from .config import config

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Generates and manages semantic embeddings for text data.
    Uses Sentence Transformers for multilingual support.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_enabled: Optional[bool] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of the Sentence Transformer model
            cache_enabled: Whether to cache embeddings
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name or config.embedding.model_name
        self.cache_enabled = cache_enabled if cache_enabled is not None else config.cache.enabled
        self.cache_path = config.paths.cache
        
        # Auto-detect device
        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        
        logger.info(f"Embedding engine initialized (model: {self.model_name}, device: {self.device})")
    
    def load_model(self) -> SentenceTransformer:
        """Load the Sentence Transformer model."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded successfully")
        return self.model
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings
        """
        self.load_model()
        
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        texts = [self._preprocess_text(t) for t in texts]
        
        # Check cache
        if self.cache_enabled:
            cached, missing_indices, missing_texts = self._check_cache(texts)
            
            if not missing_texts:
                return cached
            
            # Encode missing texts
            new_embeddings = self.model.encode(
                missing_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            # Update cache and merge results
            result = cached.copy()
            for i, idx in enumerate(missing_indices):
                result[idx] = new_embeddings[i]
                self._cache_embedding(texts[idx], new_embeddings[i])
            
            return result
        else:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
    
    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        embedding_column: str = "embedding",
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Add embeddings to a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to embed
            embedding_column: Name for the new embedding column
            batch_size: Batch size for encoding
            
        Returns:
            DataFrame with added embedding column
        """
        if df.empty:
            df[embedding_column] = []
            return df
        
        texts = df[text_column].fillna("").tolist()
        embeddings = self.encode(texts, batch_size=batch_size)
        
        df = df.copy()
        df[embedding_column] = list(embeddings)
        
        return df
    
    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        return float(cosine_similarity(embedding1, embedding2)[0, 0])
    
    def similarity_matrix(
        self,
        embeddings1: np.ndarray,
        embeddings2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate pairwise similarity matrix.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set (if None, computes self-similarity)
            
        Returns:
            Similarity matrix
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        return cosine_similarity(embeddings1, embeddings2)
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find most similar items in corpus.
        
        Args:
            query_embedding: Query embedding
            corpus_embeddings: Corpus embeddings to search
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dicts with index and score
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
        
        # Filter by threshold and sort
        indices = np.where(similarities >= threshold)[0]
        scores = similarities[indices]
        
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            results.append({
                "index": int(indices[idx]),
                "score": float(scores[idx])
            })
        
        return results
    
    def semantic_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on documents.
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of results
            threshold: Minimum similarity
            
        Returns:
            List of results with document, index, and score
        """
        query_embedding = self.encode(query)
        doc_embeddings = self.encode(documents)
        
        results = self.find_similar(
            query_embedding,
            doc_embeddings,
            top_k=top_k,
            threshold=threshold
        )
        
        for result in results:
            result["document"] = documents[result["index"]]
        
        return results
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 5
    ) -> np.ndarray:
        """
        Cluster embeddings using K-means.
        
        Args:
            embeddings: Embeddings to cluster
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels
        """
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        return labels
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding."""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Truncate if too long
        max_length = config.embedding.max_seq_length * 4  # Approximate chars
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        content = f"{self.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_cache(
        self,
        texts: List[str]
    ) -> tuple:
        """Check cache for existing embeddings."""
        cached = np.zeros((len(texts), config.embedding.dimension))
        missing_indices = []
        missing_texts = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            
            if cache_key in self._embeddings_cache:
                cached[i] = self._embeddings_cache[cache_key]
            else:
                # Check file cache
                cache_file = self.cache_path / f"{cache_key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                            if self._is_cache_valid(data):
                                cached[i] = data['embedding']
                                self._embeddings_cache[cache_key] = data['embedding']
                                continue
                    except Exception:
                        pass
                
                missing_indices.append(i)
                missing_texts.append(text)
        
        return cached, missing_indices, missing_texts
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache an embedding."""
        cache_key = self._get_cache_key(text)
        self._embeddings_cache[cache_key] = embedding
        
        # Save to file
        cache_file = self.cache_path / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embedding': embedding,
                    'timestamp': datetime.now(),
                    'model': self.model_name
                }, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def _is_cache_valid(self, data: Dict) -> bool:
        """Check if cached data is still valid."""
        if data.get('model') != self.model_name:
            return False
        
        timestamp = data.get('timestamp')
        if timestamp:
            expiry = timestamp + timedelta(days=config.cache.expiry_days)
            if datetime.now() > expiry:
                return False
        
        return True
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self._embeddings_cache.clear()
        
        for cache_file in self.cache_path.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass
        
        logger.info("Cache cleared")
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: Path,
        metadata: Optional[Dict] = None
    ):
        """Save embeddings to file."""
        data = {
            'embeddings': embeddings,
            'model': self.model_name,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: Path) -> Optional[np.ndarray]:
        """Load embeddings from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            if data.get('model') != self.model_name:
                logger.warning(f"Model mismatch: {data.get('model')} vs {self.model_name}")
            
            return data['embeddings']
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None
