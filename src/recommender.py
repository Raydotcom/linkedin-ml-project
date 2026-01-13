"""
LinkedIn Recommender Engine
===========================

Main recommendation engine that combines data loading, embeddings, and scoring.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime

from .config import config
from .data_loader import DataLoader, UserProfile, LinkedInData, PersonalData
from .embeddings import EmbeddingEngine
from .scoring import ScoringEngine, ScoringResult

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A single recommendation item."""
    id: str
    type: str  # 'job', 'contact', 'company', 'content'
    title: str
    subtitle: str
    score: float
    score_breakdown: Dict[str, float]
    details: Dict[str, Any]
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "subtitle": self.subtitle,
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "details": self.details,
            "url": self.url
        }


@dataclass
class RecommendationResult:
    """Result containing multiple recommendations."""
    recommendations: List[Recommendation]
    total_count: int
    query_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LinkedInRecommender:
    """
    Main recommendation engine for LinkedIn data.
    
    Combines NLP-based semantic matching with structured scoring
    to provide personalized recommendations.
    """
    
    def __init__(self, config_override: Optional[Any] = None):
        """Initialize the recommender."""
        self.config = config_override or config
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.embedding_engine = EmbeddingEngine()
        self.scoring_engine: Optional[ScoringEngine] = None
        
        # Data containers
        self.linkedin_data: Optional[LinkedInData] = None
        self.personal_data: Optional[PersonalData] = None
        self.user_profile: Optional[UserProfile] = None
        
        # Embeddings cache
        self._profile_embedding: Optional[np.ndarray] = None
        self._job_embeddings: Optional[np.ndarray] = None
        self._contact_embeddings: Optional[np.ndarray] = None
        self._company_embeddings: Optional[np.ndarray] = None
        
        self._loaded = False
        logger.info("LinkedInRecommender initialized")
    
    def load_data(self) -> bool:
        """Load all data and initialize engines."""
        try:
            logger.info("Loading data...")
            
            # Load all data
            self.linkedin_data, self.personal_data, self.user_profile = \
                self.data_loader.load_all()
            
            # Initialize scoring engine with user profile
            self.scoring_engine = ScoringEngine(self.user_profile)
            
            # Generate profile embedding
            profile_text = self.user_profile.get_profile_text()
            if profile_text:
                self._profile_embedding = self.embedding_engine.encode(profile_text)
            
            # Pre-compute embeddings for job offers
            if not self.personal_data.job_offers.empty:
                self._compute_job_embeddings()
            
            # Pre-compute embeddings for contacts
            if not self.linkedin_data.connections.empty:
                self._compute_contact_embeddings()
            
            # Pre-compute embeddings for target companies
            if not self.personal_data.target_companies.empty:
                self._compute_company_embeddings()
            
            self._loaded = True
            logger.info("Data loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _compute_job_embeddings(self):
        """Compute embeddings for job offers."""
        jobs_df = self.personal_data.job_offers
        
        if "combined_text" not in jobs_df.columns:
            jobs_df["combined_text"] = jobs_df.apply(
                lambda row: f"{row.get('title', '')} {row.get('company', '')} {row.get('description', '')}",
                axis=1
            )
        
        texts = jobs_df["combined_text"].tolist()
        self._job_embeddings = self.embedding_engine.encode(texts)
        
        logger.info(f"Computed embeddings for {len(texts)} job offers")
    
    def _compute_contact_embeddings(self):
        """Compute embeddings for contacts."""
        contacts_df = self.linkedin_data.connections
        
        contacts_df["profile_text"] = contacts_df.apply(
            lambda row: f"{row.get('first_name', '')} {row.get('last_name', '')} "
                       f"{row.get('position', '')} at {row.get('company', '')}",
            axis=1
        )
        
        texts = contacts_df["profile_text"].tolist()
        self._contact_embeddings = self.embedding_engine.encode(texts)
        
        logger.info(f"Computed embeddings for {len(texts)} contacts")
    
    def _compute_company_embeddings(self):
        """Compute embeddings for target companies."""
        companies_df = self.personal_data.target_companies
        
        companies_df["description_text"] = companies_df.apply(
            lambda row: f"{row.get('company_name', '')} {row.get('sector', '')} "
                       f"{row.get('notes', '')}",
            axis=1
        )
        
        texts = companies_df["description_text"].tolist()
        self._company_embeddings = self.embedding_engine.encode(texts)
        
        logger.info(f"Computed embeddings for {len(texts)} companies")
    
    def recommend_jobs(
        self,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RecommendationResult:
        """
        Get job recommendations.
        
        Args:
            top_k: Number of recommendations to return
            min_score: Minimum score threshold
            filters: Optional filters (sector, location, contract_type)
            
        Returns:
            RecommendationResult with job recommendations
        """
        start_time = datetime.now()
        
        if not self._loaded:
            self.load_data()
        
        top_k = top_k or self.config.recommendation.default_top_k_jobs
        min_score = min_score or self.config.recommendation.min_score_threshold
        
        jobs_df = self.personal_data.job_offers.copy()
        
        if jobs_df.empty:
            return RecommendationResult(
                recommendations=[],
                total_count=0,
                query_time=0,
                metadata={"message": "No job offers in database"}
            )
        
        # Apply filters
        if filters:
            jobs_df = self._apply_filters(jobs_df, filters)
        
        recommendations = []
        
        for idx, row in jobs_df.iterrows():
            # Calculate semantic similarity
            if self._profile_embedding is not None and self._job_embeddings is not None:
                job_embedding = self._job_embeddings[idx]
                semantic_sim = self.embedding_engine.similarity(
                    self._profile_embedding,
                    job_embedding
                )
            else:
                semantic_sim = 0.5
            
            # Get network connections at company
            company = row.get("company", "")
            network_connections = self._count_connections_at_company(company)
            
            # Score the job
            job_dict = row.to_dict()
            score_result = self.scoring_engine.score_job(
                job_dict,
                semantic_similarity=semantic_sim,
                network_connections=network_connections
            )
            
            if score_result.total_score >= min_score:
                recommendations.append(Recommendation(
                    id=f"job_{idx}",
                    type="job",
                    title=row.get("title", "Unknown Position"),
                    subtitle=f"{row.get('company', 'Unknown Company')} - {row.get('location', '')}",
                    score=score_result.total_score,
                    score_breakdown={
                        "semantic": score_result.semantic_score,
                        "skills": score_result.skills_score,
                        "sector": score_result.sector_score,
                        "location": score_result.location_score,
                        "network": score_result.network_score
                    },
                    details={
                        **score_result.details,
                        "description": row.get("description", "")[:300],
                        "date_added": str(row.get("date_added", "")),
                        "status": row.get("status", "")
                    },
                    url=row.get("url")
                ))
        
        # Sort by score
        recommendations.sort(key=lambda x: x.score, reverse=True)
        recommendations = recommendations[:top_k]
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return RecommendationResult(
            recommendations=recommendations,
            total_count=len(recommendations),
            query_time=query_time,
            metadata={
                "type": "jobs",
                "filters_applied": filters or {},
                "total_jobs_analyzed": len(jobs_df)
            }
        )
    
    def recommend_contacts(
        self,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> RecommendationResult:
        """
        Get contact recommendations.
        
        Args:
            top_k: Number of recommendations
            min_score: Minimum score threshold
            filters: Optional filters
            
        Returns:
            RecommendationResult with contact recommendations
        """
        start_time = datetime.now()
        
        if not self._loaded:
            self.load_data()
        
        top_k = top_k or self.config.recommendation.default_top_k_contacts
        min_score = min_score or self.config.recommendation.min_score_threshold
        
        contacts_df = self.linkedin_data.connections.copy()
        
        if contacts_df.empty:
            return RecommendationResult(
                recommendations=[],
                total_count=0,
                query_time=0,
                metadata={"message": "No connections loaded"}
            )
        
        # Merge with notes if available
        if not self.personal_data.contacts_notes.empty:
            notes_df = self.personal_data.contacts_notes
            # Try to merge on name or URL
            if "linkedin_url" in notes_df.columns and "url" in contacts_df.columns:
                contacts_df = contacts_df.merge(
                    notes_df[["linkedin_url", "notes", "priority", "last_contact"]],
                    left_on="url",
                    right_on="linkedin_url",
                    how="left"
                )
        
        # Apply filters
        if filters:
            contacts_df = self._apply_filters(contacts_df, filters)
        
        recommendations = []
        
        for idx, row in contacts_df.iterrows():
            # Calculate semantic similarity
            if self._profile_embedding is not None and self._contact_embeddings is not None:
                try:
                    contact_embedding = self._contact_embeddings[idx]
                    semantic_sim = self.embedding_engine.similarity(
                        self._profile_embedding,
                        contact_embedding
                    )
                except IndexError:
                    semantic_sim = 0.5
            else:
                semantic_sim = 0.5
            
            # Check message history
            has_messages = self._has_message_history(row)
            
            # Score the contact
            contact_dict = row.to_dict()
            score_result = self.scoring_engine.score_contact(
                contact_dict,
                semantic_similarity=semantic_sim,
                shared_connections=0,  # Would need additional data
                message_history=has_messages
            )
            
            if score_result.total_score >= min_score:
                name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
                
                recommendations.append(Recommendation(
                    id=f"contact_{idx}",
                    type="contact",
                    title=name or "Unknown",
                    subtitle=f"{row.get('position', '')} at {row.get('company', '')}",
                    score=score_result.total_score,
                    score_breakdown={
                        "semantic": score_result.semantic_score,
                        "skills": score_result.skills_score,
                        "sector": score_result.sector_score,
                        "location": score_result.location_score,
                        "network": score_result.network_score
                    },
                    details={
                        **score_result.details,
                        "connected_on": str(row.get("connected_on", "")),
                        "notes": row.get("notes", ""),
                        "last_contact": str(row.get("last_contact", ""))
                    },
                    url=row.get("url")
                ))
        
        # Sort by score
        recommendations.sort(key=lambda x: x.score, reverse=True)
        recommendations = recommendations[:top_k]
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return RecommendationResult(
            recommendations=recommendations,
            total_count=len(recommendations),
            query_time=query_time,
            metadata={
                "type": "contacts",
                "total_contacts_analyzed": len(contacts_df)
            }
        )
    
    def recommend_companies(
        self,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        include_non_targets: bool = False
    ) -> RecommendationResult:
        """
        Get company recommendations.
        
        Args:
            top_k: Number of recommendations
            min_score: Minimum score threshold
            include_non_targets: Include companies from connections
            
        Returns:
            RecommendationResult with company recommendations
        """
        start_time = datetime.now()
        
        if not self._loaded:
            self.load_data()
        
        top_k = top_k or self.config.recommendation.default_top_k_companies
        min_score = min_score or self.config.recommendation.min_score_threshold
        
        companies_df = self.personal_data.target_companies.copy()
        
        # Optionally add companies from connections
        if include_non_targets and not self.linkedin_data.connections.empty:
            connection_companies = self.linkedin_data.connections["company"].dropna().unique()
            for company in connection_companies:
                if company and company not in companies_df["company_name"].values:
                    new_row = pd.DataFrame([{
                        "company_name": company,
                        "sector": "",
                        "priority": 4,
                        "location": "",
                        "notes": "From connections"
                    }])
                    companies_df = pd.concat([companies_df, new_row], ignore_index=True)
        
        if companies_df.empty:
            return RecommendationResult(
                recommendations=[],
                total_count=0,
                query_time=0,
                metadata={"message": "No companies to recommend"}
            )
        
        recommendations = []
        
        for idx, row in companies_df.iterrows():
            company_name = row.get("company_name", "")
            
            # Calculate semantic similarity
            if self._profile_embedding is not None and idx < len(self._company_embeddings or []):
                try:
                    company_embedding = self._company_embeddings[idx]
                    semantic_sim = self.embedding_engine.similarity(
                        self._profile_embedding,
                        company_embedding
                    )
                except (IndexError, TypeError):
                    semantic_sim = 0.5
            else:
                semantic_sim = 0.5
            
            # Count connections
            network_connections = self._count_connections_at_company(company_name)
            
            # Count job openings
            job_count = self._count_jobs_at_company(company_name)
            
            # Score the company
            company_dict = row.to_dict()
            score_result = self.scoring_engine.score_company(
                company_dict,
                semantic_similarity=semantic_sim,
                network_connections=network_connections,
                job_count=job_count
            )
            
            if score_result.total_score >= min_score:
                recommendations.append(Recommendation(
                    id=f"company_{idx}",
                    type="company",
                    title=company_name,
                    subtitle=f"{row.get('sector', '')} - {row.get('location', '')}",
                    score=score_result.total_score,
                    score_breakdown={
                        "semantic": score_result.semantic_score,
                        "skills": score_result.skills_score,
                        "sector": score_result.sector_score,
                        "location": score_result.location_score,
                        "network": score_result.network_score
                    },
                    details={
                        **score_result.details,
                        "notes": row.get("notes", ""),
                        "priority": row.get("priority", 0),
                        "connections_count": network_connections,
                        "job_openings": job_count
                    },
                    url=row.get("url")
                ))
        
        # Sort by score
        recommendations.sort(key=lambda x: x.score, reverse=True)
        recommendations = recommendations[:top_k]
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return RecommendationResult(
            recommendations=recommendations,
            total_count=len(recommendations),
            query_time=query_time,
            metadata={
                "type": "companies",
                "total_companies_analyzed": len(companies_df)
            }
        )
    
    def recommend_content(
        self,
        top_k: Optional[int] = None,
        topics: Optional[List[str]] = None
    ) -> RecommendationResult:
        """
        Get content recommendations.
        
        Note: This is a placeholder that would need content data
        (posts, articles) to be truly functional.
        
        Args:
            top_k: Number of recommendations
            topics: Topics to filter by
            
        Returns:
            RecommendationResult with content recommendations
        """
        start_time = datetime.now()
        
        if not self._loaded:
            self.load_data()
        
        top_k = top_k or self.config.recommendation.default_top_k_content
        
        # Check for content data
        content_df = self.personal_data.content_interests
        
        if content_df.empty:
            # Generate topic suggestions based on profile
            suggested_topics = self._generate_topic_suggestions()
            
            return RecommendationResult(
                recommendations=[],
                total_count=0,
                query_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "message": "No content data available",
                    "suggested_topics": suggested_topics,
                    "tip": "Add content interests to data/personal/content_interests.csv"
                }
            )
        
        # TODO: Implement content recommendation when data is available
        recommendations = []
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return RecommendationResult(
            recommendations=recommendations,
            total_count=len(recommendations),
            query_time=query_time,
            metadata={"type": "content"}
        )
    
    def search(
        self,
        query: str,
        search_type: str = "all",
        top_k: int = 10
    ) -> RecommendationResult:
        """
        Semantic search across all data.
        
        Args:
            query: Search query
            search_type: 'jobs', 'contacts', 'companies', or 'all'
            top_k: Number of results
            
        Returns:
            RecommendationResult with search results
        """
        start_time = datetime.now()
        
        if not self._loaded:
            self.load_data()
        
        query_embedding = self.embedding_engine.encode(query)
        
        recommendations = []
        
        # Search jobs
        if search_type in ["jobs", "all"] and self._job_embeddings is not None:
            job_results = self.embedding_engine.find_similar(
                query_embedding,
                self._job_embeddings,
                top_k=top_k
            )
            
            for result in job_results:
                idx = result["index"]
                row = self.personal_data.job_offers.iloc[idx]
                
                recommendations.append(Recommendation(
                    id=f"job_{idx}",
                    type="job",
                    title=row.get("title", "Unknown"),
                    subtitle=row.get("company", ""),
                    score=result["score"],
                    score_breakdown={"semantic": result["score"]},
                    details={"description": row.get("description", "")[:200]},
                    url=row.get("url")
                ))
        
        # Search contacts
        if search_type in ["contacts", "all"] and self._contact_embeddings is not None:
            contact_results = self.embedding_engine.find_similar(
                query_embedding,
                self._contact_embeddings,
                top_k=top_k
            )
            
            for result in contact_results:
                idx = result["index"]
                row = self.linkedin_data.connections.iloc[idx]
                name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
                
                recommendations.append(Recommendation(
                    id=f"contact_{idx}",
                    type="contact",
                    title=name,
                    subtitle=f"{row.get('position', '')} at {row.get('company', '')}",
                    score=result["score"],
                    score_breakdown={"semantic": result["score"]},
                    details={},
                    url=row.get("url")
                ))
        
        # Sort all results by score
        recommendations.sort(key=lambda x: x.score, reverse=True)
        recommendations = recommendations[:top_k]
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return RecommendationResult(
            recommendations=recommendations,
            total_count=len(recommendations),
            query_time=query_time,
            metadata={
                "query": query,
                "search_type": search_type
            }
        )
    
    def _apply_filters(
        self,
        df: pd.DataFrame,
        filters: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply filters to a DataFrame."""
        for key, value in filters.items():
            if key in df.columns:
                if isinstance(value, list):
                    df = df[df[key].isin(value)]
                else:
                    df = df[df[key].str.contains(str(value), case=False, na=False)]
        return df
    
    def _count_connections_at_company(self, company: str) -> int:
        """Count connections at a company."""
        if not company or self.linkedin_data.connections.empty:
            return 0
        
        company_lower = company.lower()
        connections = self.linkedin_data.connections
        
        if "company" in connections.columns:
            matches = connections["company"].str.lower().str.contains(
                company_lower, na=False
            )
            return int(matches.sum())
        
        return 0
    
    def _count_jobs_at_company(self, company: str) -> int:
        """Count job openings at a company."""
        if not company or self.personal_data.job_offers.empty:
            return 0
        
        company_lower = company.lower()
        jobs = self.personal_data.job_offers
        
        if "company" in jobs.columns:
            matches = jobs["company"].str.lower().str.contains(
                company_lower, na=False
            )
            return int(matches.sum())
        
        return 0
    
    def _has_message_history(self, contact: pd.Series) -> bool:
        """Check if there's message history with a contact."""
        if self.linkedin_data.messages.empty:
            return False
        
        # Check by name or URL
        name = f"{contact.get('first_name', '')} {contact.get('last_name', '')}".strip()
        
        if "from" in self.linkedin_data.messages.columns:
            return name.lower() in self.linkedin_data.messages["from"].str.lower().values
        
        return False
    
    def _generate_topic_suggestions(self) -> List[str]:
        """Generate topic suggestions based on profile."""
        topics = []
        
        # From skills
        topics.extend(self.user_profile.skills[:5])
        
        # From target sectors
        topics.extend(self.user_profile.target_sectors)
        
        # Common data science topics
        topics.extend([
            "data science",
            "machine learning",
            "analytics",
            "career development"
        ])
        
        return list(set(topics))[:10]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard visualization."""
        if not self._loaded:
            self.load_data()
        
        return {
            "profile_summary": {
                "name": self.user_profile.name,
                "skills_count": len(self.user_profile.skills),
                "positions_count": len(self.user_profile.positions),
                "target_sectors": self.user_profile.target_sectors,
                "target_locations": self.user_profile.target_locations
            },
            "data_summary": self.data_loader.get_summary(),
            "top_skills": self.user_profile.skills[:10],
            "sector_distribution": self._get_sector_distribution(),
            "network_stats": self._get_network_stats()
        }
    
    def _get_sector_distribution(self) -> Dict[str, int]:
        """Get sector distribution of connections."""
        if self.linkedin_data.connections.empty:
            return {}
        
        if "sector" in self.linkedin_data.connections.columns:
            return self.linkedin_data.connections["sector"].value_counts().to_dict()
        
        return {}
    
    def _get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        stats = {
            "total_connections": len(self.linkedin_data.connections),
            "total_messages": len(self.linkedin_data.messages),
            "target_companies": len(self.personal_data.target_companies),
            "saved_jobs": len(self.personal_data.job_offers)
        }
        
        # Connections at target companies
        target_company_connections = 0
        for _, company in self.personal_data.target_companies.iterrows():
            target_company_connections += self._count_connections_at_company(
                company.get("company_name", "")
            )
        stats["connections_at_targets"] = target_company_connections
        
        return stats
