"""
RAG Embeddings Module
Generates embeddings for financial documents and transactions.
"""

from typing import List, Dict, Optional
import numpy as np

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = None  # Will be initialized lazily
    MODEL_LOADED = False
except ImportError:
    EMBEDDING_MODEL = None
    MODEL_LOADED = False
    print("Warning: sentence-transformers not installed. Using mock embeddings.")

from config import config


def load_embedding_model(model_name: str = None) -> Optional[SentenceTransformer]:
    """
    Load the embedding model (lazy initialization).

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded SentenceTransformer model or None if unavailable
    """
    global EMBEDDING_MODEL, MODEL_LOADED

    if MODEL_LOADED and EMBEDDING_MODEL is not None:
        return EMBEDDING_MODEL

    if model_name is None:
        model_name = config.EMBEDDING_MODEL

    try:
        EMBEDDING_MODEL = SentenceTransformer(model_name)
        MODEL_LOADED = True
        return EMBEDDING_MODEL
    except Exception as e:
        print(f"Warning: Could not load embedding model: {e}")
        print("Using mock embeddings for demonstration.")
        MODEL_LOADED = False
        return None


def get_embeddings(texts: List[str], model_name: str = None) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model_name: Optional model name override

    Returns:
        numpy array of embeddings (n_texts x embedding_dim)
    """
    model = load_embedding_model(model_name)

    if model is not None:
        try:
            embeddings = model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Warning: Embedding generation failed: {e}")
            # Fall through to mock embeddings

    # Mock embeddings for fallback
    return _mock_embeddings(texts)


def embed_text(text: str, model_name: str = None) -> np.ndarray:
    """
    Generate embedding for a single text.

    Args:
        text: Text to embed
        model_name: Optional model name override

    Returns:
        Embedding vector
    """
    embeddings = get_embeddings([text], model_name)
    return embeddings[0]


def _mock_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate mock embeddings for demonstration/testing.
    Uses a simple hash-based approach to create consistent pseudo-embeddings.

    Args:
        texts: List of texts

    Returns:
        numpy array of mock embeddings
    """
    embedding_dim = 384  # Match MiniLM dimension
    embeddings = np.zeros((len(texts), embedding_dim))

    for i, text in enumerate(texts):
        # Create a deterministic but varied embedding based on text content
        text_hash = hash(text) % (10 ** 8)
        for j in range(embedding_dim):
            # Use different hash components for each dimension
            embeddings[i, j] = ((text_hash * (j + 1) * 7 + i * 13) % 1000) / 1000.0

        # Normalize to unit length
        norm = np.linalg.norm(embeddings[i])
        if norm > 0:
            embeddings[i] /= norm

    return embeddings


def similarity_score(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Similarity score between -1 and 1
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def create_document_embeddings(documents: List[Dict]) -> List[Dict]:
    """
    Create embeddings for a list of document dictionaries.

    Each document should have at least 'content' key.

    Args:
        documents: List of document dicts with 'content' and optional 'metadata'

    Returns:
        List of documents with added 'embedding' key
    """
    if not documents:
        return []

    texts = [doc.get('content', '') for doc in documents]
    embeddings = get_embeddings(texts)

    result = []
    for i, doc in enumerate(documents):
        doc_copy = doc.copy()
        doc_copy['embedding'] = embeddings[i].tolist()
        result.append(doc_copy)

    return result


class EmbeddingService:
    """Service class for embedding operations."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None

    def get_model(self):
        """Get or load the embedding model."""
        if self.model is None:
            self.model = load_embedding_model(self.model_name)
        return self.model

    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        return embed_text(text, self.model_name).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        embeddings = get_embeddings(texts, self.model_name)
        return embeddings.tolist()

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        emb1 = embed_text(text1, self.model_name)
        emb2 = embed_text(text2, self.model_name)
        return similarity_score(emb1, emb2)


# Singleton service instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
