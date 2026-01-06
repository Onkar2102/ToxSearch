"""
embeddings.py

Prompt embedding computation using sentence-transformers for Plan A+ speciation.
Model: all-MiniLM-L6-v2 (384-dimensional, L2-normalized embeddings)
"""

import numpy as np
from typing import List, Union, Optional

from utils import get_custom_logging
from utils.device_utils import get_optimal_device

get_logger, _, _, _ = get_custom_logging()


class EmbeddingModel:
    """
    Embedding model wrapper using sentence-transformers.
    Uses all-MiniLM-L6-v2 for fast, high-quality 384-dimensional embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        self.logger = get_logger("EmbeddingModel")
        self.model_name = model_name
        self.normalize = normalize
        self.embedding_dim = 384
        
        if device is None:
            device = get_optimal_device()
        self.device = device
        
        self._model = None
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            
            test_embedding = self._model.encode("test", normalize_embeddings=self.normalize)
            self.embedding_dim = len(test_embedding)
            
            self.logger.info(f"Embedding model loaded: dim={self.embedding_dim}, device={self.device}")
            
        except ImportError as e:
            self.logger.error(f"sentence-transformers not installed: {e}")
            raise
    
    def encode(
        self,
        text: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode text(s) into L2-normalized embedding vector(s)."""
        if self._model is None:
            raise RuntimeError("Embedding model not loaded")
        
        return self._model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
    
    def __repr__(self) -> str:
        return f"EmbeddingModel(model_name='{self.model_name}', embedding_dim={self.embedding_dim})"


_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    force_reload: bool = False
) -> EmbeddingModel:
    """Get or create singleton embedding model instance."""
    global _embedding_model
    
    if _embedding_model is None or force_reload:
        _embedding_model = EmbeddingModel(model_name=model_name, device=device, normalize=True)
    
    return _embedding_model


def compute_embedding(prompt: str, model: Optional[EmbeddingModel] = None) -> np.ndarray:
    """Compute L2-normalized embedding for a single prompt."""
    if model is None:
        model = get_embedding_model()
    
    embedding = model.encode(prompt)
    if embedding.ndim > 1:
        embedding = embedding.squeeze()
    
    return embedding


def compute_embeddings_batch(
    prompts: List[str],
    model: Optional[EmbeddingModel] = None,
    batch_size: int = 64,
    show_progress: bool = False
) -> np.ndarray:
    """Compute L2-normalized embeddings for a batch of prompts."""
    if model is None:
        model = get_embedding_model()
    
    if not prompts:
        return np.array([]).reshape(0, model.embedding_dim)
    
    return model.encode(prompts, batch_size=batch_size, show_progress=show_progress)


def semantic_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    """Compute cosine distance: d(u,v) = 1 - cos(u,v) = 1 - u·v for normalized vectors."""
    cosine_similarity = np.dot(e1, e2)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    return float(1.0 - cosine_similarity)


def semantic_distances_batch(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Compute distances from query to multiple embeddings."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    cosine_similarities = embeddings @ query_embedding
    cosine_similarities = np.clip(cosine_similarities, -1.0, 1.0)
    return 1.0 - cosine_similarities


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.clip(np.dot(e1, e2), -1.0, 1.0))


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding vector."""
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

