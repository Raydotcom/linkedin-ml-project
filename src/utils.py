"""
Utility Functions Module
========================

Common utility functions for the LinkedIn Recommender.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


def normalize_text(text: str, lowercase: bool = True, remove_accents: bool = True) -> str:
    """
    Normalize text for consistent matching.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_accents: Remove accents/diacritics
        
    Returns:
        Normalized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove accents
    if remove_accents:
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
    
    return text.strip()


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    # Flatten if needed
    e1 = embedding1.flatten()
    e2 = embedding2.flatten()
    
    # Calculate cosine similarity
    dot_product = np.dot(e1, e2)
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum keyword length
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Common stop words (French + English)
    stop_words = {
        'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'en', 'à',
        'au', 'aux', 'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta',
        'tes', 'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur',
        'leurs', 'qui', 'que', 'quoi', 'dont', 'où', 'pour', 'par', 'sur',
        'avec', 'sans', 'sous', 'dans', 'entre', 'vers', 'chez', 'comme',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'we', 'you', 'they',
        'i', 'he', 'she', 'my', 'your', 'his', 'her', 'our', 'their'
    }
    
    # Tokenize
    words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text.lower())
    
    # Filter
    keywords = [
        w for w in words 
        if len(w) >= min_length and w not in stop_words
    ]
    
    # Count and sort by frequency
    word_counts = {}
    for word in keywords:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    return [w for w, _ in sorted_words[:max_keywords]]


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string with multiple format support.
    
    Args:
        date_str: Date string
        
    Returns:
        datetime object or None
    """
    if not date_str:
        return None
    
    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%b %d, %Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    
    return None


def format_score(score: float, decimals: int = 1) -> str:
    """
    Format a score for display.
    
    Args:
        score: Score value (0-1)
        decimals: Number of decimal places
        
    Returns:
        Formatted score string
    """
    percentage = score * 100
    return f"{percentage:.{decimals}f}%"


def format_time_ago(dt: datetime) -> str:
    """
    Format datetime as relative time.
    
    Args:
        dt: datetime object
        
    Returns:
        Human-readable relative time
    """
    if not dt:
        return ""
    
    now = datetime.now()
    diff = now - dt
    
    if diff < timedelta(minutes=1):
        return "À l'instant"
    elif diff < timedelta(hours=1):
        minutes = int(diff.total_seconds() / 60)
        return f"Il y a {minutes} minute{'s' if minutes > 1 else ''}"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"Il y a {hours} heure{'s' if hours > 1 else ''}"
    elif diff < timedelta(days=7):
        days = diff.days
        return f"Il y a {days} jour{'s' if days > 1 else ''}"
    elif diff < timedelta(days=30):
        weeks = diff.days // 7
        return f"Il y a {weeks} semaine{'s' if weeks > 1 else ''}"
    elif diff < timedelta(days=365):
        months = diff.days // 30
        return f"Il y a {months} mois"
    else:
        years = diff.days // 365
        return f"Il y a {years} an{'s' if years > 1 else ''}"


def generate_id(data: Any) -> str:
    """
    Generate a unique ID from data.
    
    Args:
        data: Input data (dict, string, etc.)
        
    Returns:
        MD5 hash string
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    elif not isinstance(data, str):
        data = str(data)
    
    return hashlib.md5(data.encode()).hexdigest()[:12]


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text or ""
    
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix


def merge_dicts(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def batch_process(items: List, batch_size: int = 100):
    """
    Yield batches from a list.
    
    Args:
        items: List of items
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def calculate_match_percentage(
    user_items: List[str],
    target_items: List[str],
    case_sensitive: bool = False
) -> Tuple[float, List[str]]:
    """
    Calculate percentage match between two lists.
    
    Args:
        user_items: User's items (e.g., skills)
        target_items: Target items to match
        case_sensitive: Whether to match case-sensitively
        
    Returns:
        Tuple of (match percentage, matched items)
    """
    if not target_items:
        return 0.0, []
    
    if not case_sensitive:
        user_set = {item.lower() for item in user_items}
        target_set = {item.lower() for item in target_items}
    else:
        user_set = set(user_items)
        target_set = set(target_items)
    
    matched = user_set & target_set
    percentage = len(matched) / len(target_set)
    
    return percentage, list(matched)


def create_score_breakdown_chart_data(
    score_breakdown: Dict[str, float],
    labels: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Create data for a score breakdown chart.
    
    Args:
        score_breakdown: Dictionary of component scores
        labels: Optional custom labels
        
    Returns:
        List of chart data points
    """
    default_labels = {
        "semantic": "Similarité sémantique",
        "skills": "Compétences",
        "sector": "Secteur",
        "location": "Localisation",
        "network": "Réseau"
    }
    
    labels = labels or default_labels
    
    chart_data = []
    for key, value in score_breakdown.items():
        chart_data.append({
            "category": labels.get(key, key),
            "value": round(value * 100, 1),
            "key": key
        })
    
    return chart_data


def validate_csv_structure(
    df,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate CSV structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        optional_columns: List of optional column names
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    df_columns = [col.lower() for col in df.columns]
    required_lower = [col.lower() for col in required_columns]
    
    missing = [col for col in required_lower if col not in df_columns]
    
    return len(missing) == 0, missing


def get_color_for_score(score: float) -> str:
    """
    Get a color based on score value.
    
    Args:
        score: Score value (0-1)
        
    Returns:
        Hex color code
    """
    if score >= 0.8:
        return "#22c55e"  # Green
    elif score >= 0.6:
        return "#84cc16"  # Lime
    elif score >= 0.4:
        return "#eab308"  # Yellow
    elif score >= 0.2:
        return "#f97316"  # Orange
    else:
        return "#ef4444"  # Red


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails
        
    Returns:
        Result of division or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


# Logging setup helper
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Log level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)
