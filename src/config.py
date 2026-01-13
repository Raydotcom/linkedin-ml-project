"""
Configuration module for LinkedIn Smart Recommender.
Loads settings from environment variables and provides defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class PathConfig:
    """Configuration for data paths."""
    linkedin_data: Path = field(default_factory=lambda: DATA_DIR / "linkedin")
    personal_data: Path = field(default_factory=lambda: DATA_DIR / "personal")
    cache: Path = field(default_factory=lambda: DATA_DIR / "cache")
    
    def __post_init__(self):
        # Create directories if they don't exist
        for path in [self.linkedin_data, self.personal_data, self.cache]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class EmbeddingConfig:
    """Configuration for NLP embeddings."""
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    batch_size: int = 32
    max_seq_length: int = 256
    
    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", cls.model_name),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", cls.dimension))
        )


@dataclass
class ScoringWeights:
    """Weights for recommendation scoring."""
    semantic_similarity: float = 0.35
    skills_match: float = 0.25
    sector_match: float = 0.20
    location_match: float = 0.10
    network_proximity: float = 0.10
    
    def __post_init__(self):
        # Normalize weights to sum to 1.0
        total = (
            self.semantic_similarity + 
            self.skills_match + 
            self.sector_match + 
            self.location_match + 
            self.network_proximity
        )
        if abs(total - 1.0) > 0.01:
            self.semantic_similarity /= total
            self.skills_match /= total
            self.sector_match /= total
            self.location_match /= total
            self.network_proximity /= total
    
    @classmethod
    def from_env(cls) -> "ScoringWeights":
        return cls(
            semantic_similarity=float(os.getenv("WEIGHT_SEMANTIC_SIMILARITY", 0.35)),
            skills_match=float(os.getenv("WEIGHT_SKILLS_MATCH", 0.25)),
            sector_match=float(os.getenv("WEIGHT_SECTOR_MATCH", 0.20)),
            location_match=float(os.getenv("WEIGHT_LOCATION_MATCH", 0.10)),
            network_proximity=float(os.getenv("WEIGHT_NETWORK_PROXIMITY", 0.10))
        )


@dataclass
class RecommendationConfig:
    """Configuration for recommendations."""
    default_top_k_jobs: int = 10
    default_top_k_contacts: int = 20
    default_top_k_companies: int = 15
    default_top_k_content: int = 10
    min_score_threshold: float = 0.3
    
    @classmethod
    def from_env(cls) -> "RecommendationConfig":
        return cls(
            default_top_k_jobs=int(os.getenv("DEFAULT_TOP_K_JOBS", 10)),
            default_top_k_contacts=int(os.getenv("DEFAULT_TOP_K_CONTACTS", 20)),
            default_top_k_companies=int(os.getenv("DEFAULT_TOP_K_COMPANIES", 15)),
            default_top_k_content=int(os.getenv("DEFAULT_TOP_K_CONTENT", 10)),
            min_score_threshold=float(os.getenv("MIN_SCORE_THRESHOLD", 0.3))
        )


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    expiry_days: int = 7
    
    @classmethod
    def from_env(cls) -> "CacheConfig":
        return cls(
            enabled=os.getenv("ENABLE_CACHE", "true").lower() == "true",
            expiry_days=int(os.getenv("CACHE_EXPIRY_DAYS", 7))
        )


@dataclass
class LinkedInFilesConfig:
    """Expected LinkedIn export file names."""
    connections: str = "Connections.csv"
    skills: str = "Skills.csv"
    positions: str = "Positions.csv"
    messages: str = "Messages.csv"
    profile: str = "Profile.csv"
    education: str = "Education.csv"
    
    def get_all_files(self) -> List[str]:
        return [
            self.connections,
            self.skills,
            self.positions,
            self.messages,
            self.profile,
            self.education
        ]


@dataclass
class PersonalFilesConfig:
    """Expected personal data file names."""
    target_companies: str = "target_companies.csv"
    job_offers: str = "job_offers.csv"
    preferences: str = "preferences.csv"
    contacts_notes: str = "contacts_notes.csv"
    content_interests: str = "content_interests.csv"
    
    def get_all_files(self) -> List[str]:
        return [
            self.target_companies,
            self.job_offers,
            self.preferences,
            self.contacts_notes,
            self.content_interests
        ]


@dataclass
class AppConfig:
    """Main application configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig.from_env)
    scoring: ScoringWeights = field(default_factory=ScoringWeights.from_env)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig.from_env)
    cache: CacheConfig = field(default_factory=CacheConfig.from_env)
    linkedin_files: LinkedInFilesConfig = field(default_factory=LinkedInFilesConfig)
    personal_files: PersonalFilesConfig = field(default_factory=PersonalFilesConfig)
    language: str = "fr"
    theme: str = "light"
    
    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration from environment."""
        return cls(
            language=os.getenv("APP_LANGUAGE", "fr"),
            theme=os.getenv("STREAMLIT_THEME", "light")
        )


# Global configuration instance
config = AppConfig.load()


# Sector mappings for normalization
SECTOR_MAPPINGS = {
    # Sports & Media
    "sports": ["sport", "sports", "sportif", "athletic"],
    "media": ["media", "médias", "broadcasting", "diffusion", "entertainment"],
    "tech": ["technology", "tech", "technologie", "software", "digital"],
    
    # Finance
    "banking": ["bank", "banque", "banking", "bancaire"],
    "insurance": ["insurance", "assurance", "insurtech"],
    "finance": ["finance", "financial", "financier", "investment"],
    
    # Retail & Consumer
    "retail": ["retail", "commerce", "e-commerce", "ecommerce"],
    "consumer": ["consumer goods", "fmcg", "grande consommation"],
    
    # Consulting & Services
    "consulting": ["consulting", "conseil", "advisory"],
    "services": ["services", "professional services"],
}

# Location mappings for normalization
LOCATION_MAPPINGS = {
    "paris": ["paris", "île-de-france", "idf", "92", "75", "93", "94"],
    "lyon": ["lyon", "rhône", "69"],
    "remote": ["remote", "télétravail", "à distance", "full remote"],
    "france": ["france", "fr", "national"],
}

# Contract type mappings
CONTRACT_MAPPINGS = {
    "stage": ["stage", "internship", "intern", "stagiaire"],
    "alternance": ["alternance", "apprentissage", "apprentice"],
    "cdi": ["cdi", "permanent", "full-time", "temps plein"],
    "cdd": ["cdd", "contract", "fixed-term", "temporary"],
}
