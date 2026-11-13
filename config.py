# FinanceIQ Configuration

from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    """Application configuration settings."""

    # Transaction categories
    CATEGORIES: List[str] = field(default_factory=lambda: [
        "Food and Dining",
        "Shopping",
        "Transportation",
        "Groceries",
        "Entertainment",
        "Healthcare",
        "Education",
        "Bills and Utilities",
        "Rent",
        "Travel",
        "Salary and Income",
        "Transfers",
        "Other"
    ])

    # Date format patterns
    DATE_FORMATS: List[str] = field(default_factory=lambda: [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d/%m/%y",
        "%d-%m-%y"
    ])

    # ChromaDB settings
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "finance_transactions"

    # Embedding model
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking settings for RAG
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # LLM settings (defaults, overridden by .env)
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 2048

    # Color scheme for the UI
    COLORS = {
        "background": "#0f172a",
        "card_background": "#1e293b",
        "text_primary": "#ffffff",
        "text_secondary": "#94a3b8",
        "income": "#22c55e",
        "expense": "#ef4444",
        "primary": "#3b82f6",
        "secondary": "#6366f1",
        "accent": "#06b6d4",
        "border": "#334155"
    }


# Singleton config instance
config = Config()
