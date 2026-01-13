"""
Scoring Engine Module
=====================

Multi-criteria scoring system for recommendations.
Combines semantic similarity with structured matching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from rapidfuzz import fuzz
import re

from .config import config, SECTOR_MAPPINGS, LOCATION_MAPPINGS, CONTRACT_MAPPINGS
from .data_loader import UserProfile

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result of scoring an item."""
    total_score: float
    semantic_score: float
    skills_score: float
    sector_score: float
    location_score: float
    network_score: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "semantic_score": self.semantic_score,
            "skills_score": self.skills_score,
            "sector_score": self.sector_score,
            "location_score": self.location_score,
            "network_score": self.network_score,
            "details": self.details
        }


class ScoringEngine:
    """
    Multi-criteria scoring engine for recommendations.
    Combines multiple signals into a weighted final score.
    """
    
    def __init__(
        self,
        user_profile: UserProfile,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the scoring engine.
        
        Args:
            user_profile: User's profile and preferences
            weights: Custom weights for scoring components
        """
        self.user_profile = user_profile
        
        # Set weights from config or custom
        if weights:
            self.weights = weights
        else:
            self.weights = {
                "semantic": config.scoring.semantic_similarity,
                "skills": config.scoring.skills_match,
                "sector": config.scoring.sector_match,
                "location": config.scoring.location_match,
                "network": config.scoring.network_proximity
            }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Build matching indices
        self._build_skill_index()
        self._build_sector_index()
        self._build_location_index()
        
        logger.info(f"Scoring engine initialized with weights: {self.weights}")
    
    def _build_skill_index(self):
        """Build index for skill matching."""
        self.user_skills = set()
        for skill in self.user_profile.skills:
            normalized = skill.lower().strip()
            self.user_skills.add(normalized)
            
            # Add variations
            self.user_skills.add(normalized.replace("-", " "))
            self.user_skills.add(normalized.replace(" ", "-"))
    
    def _build_sector_index(self):
        """Build index for sector matching."""
        self.target_sectors = set()
        for sector in self.user_profile.target_sectors:
            normalized = sector.lower().strip()
            self.target_sectors.add(normalized)
            
            # Add mappings
            for key, variants in SECTOR_MAPPINGS.items():
                if normalized in variants or normalized == key:
                    self.target_sectors.add(key)
                    self.target_sectors.update(variants)
    
    def _build_location_index(self):
        """Build index for location matching."""
        self.target_locations = set()
        for location in self.user_profile.target_locations:
            normalized = location.lower().strip()
            self.target_locations.add(normalized)
            
            # Add mappings
            for key, variants in LOCATION_MAPPINGS.items():
                if normalized in variants or normalized == key:
                    self.target_locations.add(key)
                    self.target_locations.update(variants)
    
    def score_job(
        self,
        job: Dict[str, Any],
        semantic_similarity: float = 0.0,
        network_connections: int = 0
    ) -> ScoringResult:
        """
        Score a job offer.
        
        Args:
            job: Job offer data
            semantic_similarity: Pre-computed semantic similarity (0-1)
            network_connections: Number of network connections at company
            
        Returns:
            ScoringResult with component scores
        """
        details = {}
        
        # Semantic score
        semantic_score = semantic_similarity
        details["semantic"] = {"similarity": semantic_similarity}
        
        # Skills score
        job_skills = self._extract_skills(job.get("description", "") + " " + job.get("requirements", ""))
        skills_score, matched_skills = self._calculate_skills_match(job_skills)
        details["skills"] = {"matched": matched_skills, "total_job_skills": len(job_skills)}
        
        # Sector score
        job_sector = job.get("sector", "") or self._infer_sector(job.get("company", ""))
        sector_score = self._calculate_sector_match(job_sector)
        details["sector"] = {"job_sector": job_sector, "match": sector_score > 0}
        
        # Location score
        job_location = job.get("location", "")
        location_score = self._calculate_location_match(job_location)
        details["location"] = {"job_location": job_location, "match": location_score > 0}
        
        # Network score
        network_score = self._calculate_network_score(network_connections)
        details["network"] = {"connections": network_connections}
        
        # Contract type bonus
        contract_bonus = 0.0
        job_contract = job.get("contract_type", "").lower()
        if job_contract:
            for target_type in self.user_profile.target_contract_types:
                if self._match_contract_type(job_contract, target_type):
                    contract_bonus = 0.1
                    details["contract_match"] = True
                    break
        
        # Calculate total score
        total_score = (
            self.weights["semantic"] * semantic_score +
            self.weights["skills"] * skills_score +
            self.weights["sector"] * sector_score +
            self.weights["location"] * location_score +
            self.weights["network"] * network_score +
            contract_bonus
        )
        
        # Apply priority boost if company is in target list
        priority = job.get("priority", 0)
        if priority:
            priority_boost = (4 - min(priority, 3)) * 0.05  # 0.05 to 0.15
            total_score += priority_boost
            details["priority_boost"] = priority_boost
        
        # Normalize to 0-1
        total_score = min(max(total_score, 0.0), 1.0)
        
        return ScoringResult(
            total_score=total_score,
            semantic_score=semantic_score,
            skills_score=skills_score,
            sector_score=sector_score,
            location_score=location_score,
            network_score=network_score,
            details=details
        )
    
    def score_contact(
        self,
        contact: Dict[str, Any],
        semantic_similarity: float = 0.0,
        shared_connections: int = 0,
        message_history: bool = False
    ) -> ScoringResult:
        """
        Score a contact for recommendation.
        
        Args:
            contact: Contact data
            semantic_similarity: Profile similarity
            shared_connections: Number of shared connections
            message_history: Whether there's message history
            
        Returns:
            ScoringResult
        """
        details = {}
        
        # Semantic score (based on profile/headline match)
        semantic_score = semantic_similarity
        details["semantic"] = {"similarity": semantic_similarity}
        
        # Skills score (inferred from position)
        position = contact.get("position", "")
        inferred_skills = self._infer_skills_from_position(position)
        skills_score, matched = self._calculate_skills_match(inferred_skills)
        details["skills"] = {"inferred": inferred_skills[:5], "matched": matched}
        
        # Sector score
        company = contact.get("company", "")
        sector = self._infer_sector(company)
        sector_score = self._calculate_sector_match(sector)
        details["sector"] = {"company": company, "inferred_sector": sector}
        
        # Location score
        location = contact.get("location", "")
        location_score = self._calculate_location_match(location)
        details["location"] = {"location": location}
        
        # Network score (shared connections + message history)
        network_score = self._calculate_network_score(shared_connections)
        if message_history:
            network_score = min(network_score + 0.2, 1.0)
            details["has_message_history"] = True
        details["network"] = {"shared_connections": shared_connections}
        
        # Seniority bonus
        seniority = contact.get("seniority", "") or self._infer_seniority(position)
        seniority_bonus = self._calculate_seniority_bonus(seniority)
        details["seniority"] = {"level": seniority, "bonus": seniority_bonus}
        
        # Priority from notes
        priority = contact.get("priority", 0)
        if priority:
            priority_boost = (4 - min(priority, 3)) * 0.1
            details["priority_boost"] = priority_boost
        else:
            priority_boost = 0
        
        # Total score
        total_score = (
            self.weights["semantic"] * semantic_score +
            self.weights["skills"] * skills_score +
            self.weights["sector"] * sector_score +
            self.weights["location"] * location_score +
            self.weights["network"] * network_score +
            seniority_bonus +
            priority_boost
        )
        
        total_score = min(max(total_score, 0.0), 1.0)
        
        return ScoringResult(
            total_score=total_score,
            semantic_score=semantic_score,
            skills_score=skills_score,
            sector_score=sector_score,
            location_score=location_score,
            network_score=network_score,
            details=details
        )
    
    def score_company(
        self,
        company: Dict[str, Any],
        semantic_similarity: float = 0.0,
        network_connections: int = 0,
        job_count: int = 0
    ) -> ScoringResult:
        """
        Score a company for recommendation.
        
        Args:
            company: Company data
            semantic_similarity: Description similarity
            network_connections: Connections at company
            job_count: Number of relevant job openings
            
        Returns:
            ScoringResult
        """
        details = {}
        
        # Semantic score
        semantic_score = semantic_similarity
        
        # Sector score
        sector = company.get("sector", "")
        sector_score = self._calculate_sector_match(sector)
        details["sector"] = {"sector": sector}
        
        # Location score
        location = company.get("location", "")
        location_score = self._calculate_location_match(location)
        details["location"] = {"location": location}
        
        # Network score
        network_score = self._calculate_network_score(network_connections)
        details["network"] = {"connections": network_connections}
        
        # Job availability bonus
        job_bonus = min(job_count * 0.05, 0.2)
        details["job_openings"] = {"count": job_count, "bonus": job_bonus}
        
        # Priority
        priority = company.get("priority", 0)
        if priority:
            priority_boost = (4 - min(priority, 3)) * 0.15
            details["priority_boost"] = priority_boost
        else:
            priority_boost = 0
        
        # Skills score (placeholder - would need company tech stack data)
        skills_score = 0.5  # Neutral
        
        total_score = (
            self.weights["semantic"] * semantic_score +
            self.weights["skills"] * skills_score +
            self.weights["sector"] * sector_score +
            self.weights["location"] * location_score +
            self.weights["network"] * network_score +
            job_bonus +
            priority_boost
        )
        
        total_score = min(max(total_score, 0.0), 1.0)
        
        return ScoringResult(
            total_score=total_score,
            semantic_score=semantic_score,
            skills_score=skills_score,
            sector_score=sector_score,
            location_score=location_score,
            network_score=network_score,
            details=details
        )
    
    def score_content(
        self,
        content: Dict[str, Any],
        semantic_similarity: float = 0.0,
        author_score: float = 0.0,
        engagement: int = 0
    ) -> ScoringResult:
        """
        Score content for recommendation.
        
        Args:
            content: Content/post data
            semantic_similarity: Topic similarity
            author_score: Score of the author
            engagement: Engagement metrics
            
        Returns:
            ScoringResult
        """
        details = {}
        
        # Semantic score (topic relevance)
        semantic_score = semantic_similarity
        details["topic_relevance"] = semantic_similarity
        
        # Author score contribution
        author_weight = 0.3
        author_component = author_score * author_weight
        details["author"] = {"score": author_score}
        
        # Engagement normalization (log scale)
        engagement_score = min(np.log1p(engagement) / 10, 1.0)
        details["engagement"] = {"raw": engagement, "normalized": engagement_score}
        
        # Sector relevance
        topic = content.get("topic", "")
        sector_score = self._calculate_sector_match(topic)
        
        # Skills mention
        text = content.get("text", "")
        skills_in_content = self._extract_skills(text)
        skills_score, matched = self._calculate_skills_match(skills_in_content)
        details["skills_mentioned"] = matched
        
        total_score = (
            0.4 * semantic_score +
            0.2 * skills_score +
            0.15 * sector_score +
            0.15 * author_component +
            0.1 * engagement_score
        )
        
        return ScoringResult(
            total_score=min(max(total_score, 0.0), 1.0),
            semantic_score=semantic_score,
            skills_score=skills_score,
            sector_score=sector_score,
            location_score=0.0,
            network_score=author_component,
            details=details
        )
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text."""
        if not text:
            return []
        
        text_lower = text.lower()
        found_skills = []
        
        # Common data science / analytics skills
        skill_patterns = [
            r'\bpython\b', r'\bsql\b', r'\br\b(?!\s*&)', r'\bsas\b',
            r'\bmachine learning\b', r'\bml\b', r'\bdeep learning\b',
            r'\bdata analysis\b', r'\bdata science\b', r'\banalytics\b',
            r'\btableau\b', r'\bpower bi\b', r'\bexcel\b',
            r'\bstatistics\b', r'\bstatistiques\b',
            r'\bpandas\b', r'\bnumpy\b', r'\bscikit-learn\b',
            r'\btensorflow\b', r'\bpytorch\b', r'\bkeras\b',
            r'\bnlp\b', r'\bnatural language\b',
            r'\bspark\b', r'\bhadoop\b', r'\bbig data\b',
            r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bcloud\b',
            r'\bdocker\b', r'\bkubernetes\b', r'\bgit\b',
            r'\bapi\b', r'\brest\b', r'\betl\b',
            r'\bvisualization\b', r'\bvisualisation\b',
            r'\beconometrics\b', r'\béconométrie\b',
            r'\bregression\b', r'\bclassification\b',
            r'\bclustering\b', r'\btime series\b',
        ]
        
        for pattern in skill_patterns:
            if re.search(pattern, text_lower):
                # Extract the matched skill
                match = re.search(pattern, text_lower)
                if match:
                    found_skills.append(match.group().strip())
        
        return list(set(found_skills))
    
    def _calculate_skills_match(self, job_skills: List[str]) -> Tuple[float, List[str]]:
        """Calculate skills match score."""
        if not job_skills or not self.user_skills:
            return 0.5, []  # Neutral score if no data
        
        matched = []
        for skill in job_skills:
            skill_lower = skill.lower().strip()
            
            # Direct match
            if skill_lower in self.user_skills:
                matched.append(skill)
                continue
            
            # Fuzzy match
            for user_skill in self.user_skills:
                if fuzz.ratio(skill_lower, user_skill) > 85:
                    matched.append(skill)
                    break
        
        if not job_skills:
            return 0.5, matched
        
        score = len(matched) / len(job_skills)
        return score, matched
    
    def _calculate_sector_match(self, sector: str) -> float:
        """Calculate sector match score."""
        if not sector or not self.target_sectors:
            return 0.5
        
        sector_lower = sector.lower().strip()
        
        # Direct match
        if sector_lower in self.target_sectors:
            return 1.0
        
        # Check variants
        for key, variants in SECTOR_MAPPINGS.items():
            if sector_lower in variants or key in sector_lower:
                if key in self.target_sectors or any(v in self.target_sectors for v in variants):
                    return 1.0
        
        # Fuzzy match
        for target in self.target_sectors:
            if fuzz.partial_ratio(sector_lower, target) > 80:
                return 0.8
        
        return 0.3
    
    def _calculate_location_match(self, location: str) -> float:
        """Calculate location match score."""
        if not location:
            return 0.5
        
        if not self.target_locations:
            return 0.5
        
        location_lower = location.lower().strip()
        
        # Check for remote
        if "remote" in self.target_locations or "télétravail" in self.target_locations:
            if any(r in location_lower for r in ["remote", "télétravail", "à distance"]):
                return 1.0
        
        # Direct match
        if location_lower in self.target_locations:
            return 1.0
        
        # Check variants
        for key, variants in LOCATION_MAPPINGS.items():
            if any(v in location_lower for v in variants):
                if key in self.target_locations or any(v in self.target_locations for v in variants):
                    return 1.0
        
        # Partial match (same country)
        if "france" in self.target_locations and any(
            fr in location_lower for fr in ["france", "paris", "lyon", "marseille", "toulouse"]
        ):
            return 0.7
        
        return 0.3
    
    def _calculate_network_score(self, connections: int) -> float:
        """Calculate network proximity score."""
        if connections == 0:
            return 0.0
        elif connections == 1:
            return 0.4
        elif connections <= 3:
            return 0.6
        elif connections <= 10:
            return 0.8
        else:
            return 1.0
    
    def _calculate_seniority_bonus(self, seniority: str) -> float:
        """Calculate bonus based on contact seniority."""
        seniority_bonuses = {
            "c-level": 0.15,
            "director": 0.12,
            "manager": 0.08,
            "senior": 0.05,
            "mid": 0.02,
            "junior": 0.0
        }
        return seniority_bonuses.get(seniority.lower(), 0.0)
    
    def _infer_sector(self, company: str) -> str:
        """Infer sector from company name."""
        if not company:
            return ""
        
        company_lower = company.lower()
        
        known_companies = {
            "canal": "media", "tf1": "media", "m6": "media",
            "nike": "sports", "adidas": "sports", "decathlon": "sports",
            "bnp": "banking", "société générale": "banking", "crédit agricole": "banking",
            "axa": "insurance", "allianz": "insurance",
            "google": "tech", "microsoft": "tech", "amazon": "tech", "meta": "tech",
            "mckinsey": "consulting", "bcg": "consulting", "bain": "consulting",
        }
        
        for key, sector in known_companies.items():
            if key in company_lower:
                return sector
        
        return ""
    
    def _infer_seniority(self, title: str) -> str:
        """Infer seniority from job title."""
        if not title:
            return "unknown"
        
        title_lower = title.lower()
        
        if any(t in title_lower for t in ["ceo", "cto", "cfo", "chief", "president"]):
            return "c-level"
        if any(t in title_lower for t in ["director", "directeur", "vp", "vice president"]):
            return "director"
        if any(t in title_lower for t in ["manager", "head of", "responsable", "lead"]):
            return "manager"
        if any(t in title_lower for t in ["senior", "sr.", "principal"]):
            return "senior"
        if any(t in title_lower for t in ["junior", "jr.", "stagiaire", "intern"]):
            return "junior"
        
        return "mid"
    
    def _infer_skills_from_position(self, position: str) -> List[str]:
        """Infer likely skills from job position."""
        if not position:
            return []
        
        position_lower = position.lower()
        inferred = []
        
        # Data roles
        if any(t in position_lower for t in ["data", "analyst", "analytics"]):
            inferred.extend(["sql", "python", "analytics", "excel"])
        
        if any(t in position_lower for t in ["data scientist", "machine learning", "ml"]):
            inferred.extend(["python", "machine learning", "statistics"])
        
        if any(t in position_lower for t in ["engineer", "developer", "développeur"]):
            inferred.extend(["python", "git", "sql"])
        
        if "recruiter" in position_lower or "rh" in position_lower or "hr" in position_lower:
            inferred.append("recrutement")
        
        return list(set(inferred))
    
    def _match_contract_type(self, job_contract: str, target: str) -> bool:
        """Check if contract type matches."""
        job_lower = job_contract.lower()
        target_lower = target.lower()
        
        for key, variants in CONTRACT_MAPPINGS.items():
            if target_lower in variants or target_lower == key:
                if any(v in job_lower for v in variants) or key in job_lower:
                    return True
        
        return False
