"""
Application Configuration
Manages all configuration settings from environment variables or Streamlit secrets.
"""
import os
from pathlib import Path
from typing import Optional

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """Get secret from Streamlit secrets or environment variables."""
    if HAS_STREAMLIT:
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except Exception:
            return os.getenv(key, default)
    return os.getenv(key, default)


class Config:
    """Application configuration class."""
    
    # Base paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    # LLM Configuration
    LLM_PROVIDER: str = get_secret("LLM_PROVIDER", "groq")
    GROQ_API_KEY: str = get_secret("GROQ_API_KEY", "")
    OLLAMA_BASE_URL: str = get_secret("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = get_secret("LLM_MODEL", "llama-3.1-8b-instant")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = get_secret("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384  # Dimension for all-MiniLM-L6-v2
    
    # Database Configuration
    DATABASE_URL: str = get_secret("DATABASE_URL", f"sqlite:///{DATA_DIR}/app.db")
    SECRET_KEY: str = get_secret("SECRET_KEY", "change-this-in-production")
    
    # Endee Vector Database
    ENDEE_PATH: str = get_secret("ENDEE_PATH", str(BASE_DIR / "endee_db"))
    ENDEE_SERVER_URL: str = get_secret("ENDEE_SERVER_URL", "http://localhost:8080")
    ENDEE_AUTH_TOKEN: str = get_secret("ENDEE_AUTH_TOKEN", "")
    
    # Application Settings
    MAX_UPLOAD_SIZE_MB: int = int(get_secret("MAX_UPLOAD_SIZE_MB", "200"))
    CHUNK_SIZE: int = int(get_secret("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(get_secret("CHUNK_OVERLAP", "50"))
    TOP_K_RETRIEVAL: int = int(get_secret("TOP_K_RETRIEVAL", "3"))
    
    # JWT Settings
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Agent Settings
    MAX_AGENT_STEPS: int = 10
    AGENT_TIMEOUT_SECONDS: int = 300
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Path(cls.ENDEE_PATH).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_llm_client(cls):
        """Get the appropriate LLM client based on configuration."""
        if cls.LLM_PROVIDER == "groq":
            from groq import Groq
            return Groq(api_key=cls.GROQ_API_KEY)
        elif cls.LLM_PROVIDER == "ollama":
            import ollama
            return ollama
        else:
            raise ValueError(f"Unsupported LLM provider: {cls.LLM_PROVIDER}")


# Create config instance
config = Config()
