"""
Data Loader Module
==================

Handles loading and preprocessing of LinkedIn exports and personal data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
import re

from .config import config, SECTOR_MAPPINGS, LOCATION_MAPPINGS, CONTRACT_MAPPINGS

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Represents the user's LinkedIn profile and preferences."""
    name: str = ""
    headline: str = ""
    summary: str = ""
    skills: List[str] = field(default_factory=list)
    positions: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    target_sectors: List[str] = field(default_factory=list)
    target_locations: List[str] = field(default_factory=list)
    target_contract_types: List[str] = field(default_factory=list)
    preferences: Dict[str, float] = field(default_factory=dict)
    
    def get_profile_text(self) -> str:
        """Generate a text representation of the profile for embedding."""
        parts = []
        
        if self.headline:
            parts.append(self.headline)
        
        if self.summary:
            parts.append(self.summary)
        
        if self.skills:
            parts.append("Compétences: " + ", ".join(self.skills[:20]))
        
        for pos in self.positions[:3]:
            pos_text = f"{pos.get('title', '')} chez {pos.get('company', '')}"
            if pos.get('description'):
                pos_text += f": {pos.get('description', '')[:200]}"
            parts.append(pos_text)
        
        return " ".join(parts)


@dataclass
class LinkedInData:
    """Container for all LinkedIn data."""
    connections: pd.DataFrame = field(default_factory=pd.DataFrame)
    skills: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    messages: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    education: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class PersonalData:
    """Container for all personal data."""
    target_companies: pd.DataFrame = field(default_factory=pd.DataFrame)
    job_offers: pd.DataFrame = field(default_factory=pd.DataFrame)
    preferences: pd.DataFrame = field(default_factory=pd.DataFrame)
    contacts_notes: pd.DataFrame = field(default_factory=pd.DataFrame)
    content_interests: pd.DataFrame = field(default_factory=pd.DataFrame)


class DataLoader:
    """
    Loads and preprocesses data from LinkedIn exports and personal CSV files.
    """
    
    def __init__(self, config_override: Optional[Any] = None):
        """Initialize the data loader."""
        self.config = config_override or config
        self.linkedin_data = LinkedInData()
        self.personal_data = PersonalData()
        self.user_profile = UserProfile()
        self._loaded = False
    
    def load_all(self) -> Tuple[LinkedInData, PersonalData, UserProfile]:
        """Load all data sources."""
        logger.info("Loading all data sources...")
        
        self.load_linkedin_data()
        self.load_personal_data()
        self.build_user_profile()
        
        self._loaded = True
        logger.info("All data loaded successfully")
        
        return self.linkedin_data, self.personal_data, self.user_profile
    
    def load_linkedin_data(self) -> LinkedInData:
        """Load LinkedIn export files."""
        linkedin_path = self.config.paths.linkedin_data
        files_config = self.config.linkedin_files
        
        # Load Connections
        connections_file = linkedin_path / files_config.connections
        if connections_file.exists():
            self.linkedin_data.connections = self._load_csv(
                connections_file,
                processors=[self._process_connections]
            )
            logger.info(f"Loaded {len(self.linkedin_data.connections)} connections")
        
        # Load Skills
        skills_file = linkedin_path / files_config.skills
        if skills_file.exists():
            self.linkedin_data.skills = self._load_csv(
                skills_file,
                processors=[self._process_skills]
            )
            logger.info(f"Loaded {len(self.linkedin_data.skills)} skills")
        
        # Load Positions
        positions_file = linkedin_path / files_config.positions
        if positions_file.exists():
            self.linkedin_data.positions = self._load_csv(
                positions_file,
                processors=[self._process_positions]
            )
            logger.info(f"Loaded {len(self.linkedin_data.positions)} positions")
        
        # Load Messages (optional)
        messages_file = linkedin_path / files_config.messages
        if messages_file.exists():
            self.linkedin_data.messages = self._load_csv(
                messages_file,
                processors=[self._process_messages]
            )
            logger.info(f"Loaded {len(self.linkedin_data.messages)} messages")
        
        # Load Profile
        profile_file = linkedin_path / files_config.profile
        if profile_file.exists():
            self.linkedin_data.profile = self._load_csv(profile_file)
            logger.info("Loaded profile data")
        
        # Load Education
        education_file = linkedin_path / files_config.education
        if education_file.exists():
            self.linkedin_data.education = self._load_csv(education_file)
            logger.info(f"Loaded {len(self.linkedin_data.education)} education entries")
        
        return self.linkedin_data
    
    def load_personal_data(self) -> PersonalData:
        """Load personal data files."""
        personal_path = self.config.paths.personal_data
        files_config = self.config.personal_files
        
        # Load Target Companies
        companies_file = personal_path / files_config.target_companies
        if companies_file.exists():
            self.personal_data.target_companies = self._load_csv(
                companies_file,
                processors=[self._process_target_companies]
            )
            logger.info(f"Loaded {len(self.personal_data.target_companies)} target companies")
        
        # Load Job Offers
        jobs_file = personal_path / files_config.job_offers
        if jobs_file.exists():
            self.personal_data.job_offers = self._load_csv(
                jobs_file,
                processors=[self._process_job_offers]
            )
            logger.info(f"Loaded {len(self.personal_data.job_offers)} job offers")
        
        # Load Preferences
        prefs_file = personal_path / files_config.preferences
        if prefs_file.exists():
            self.personal_data.preferences = self._load_csv(
                prefs_file,
                processors=[self._process_preferences]
            )
            logger.info(f"Loaded {len(self.personal_data.preferences)} preferences")
        
        # Load Contact Notes
        contacts_file = personal_path / files_config.contacts_notes
        if contacts_file.exists():
            self.personal_data.contacts_notes = self._load_csv(
                contacts_file,
                processors=[self._process_contacts_notes]
            )
            logger.info(f"Loaded {len(self.personal_data.contacts_notes)} contact notes")
        
        # Load Content Interests
        content_file = personal_path / files_config.content_interests
        if content_file.exists():
            self.personal_data.content_interests = self._load_csv(content_file)
            logger.info(f"Loaded {len(self.personal_data.content_interests)} content interests")
        
        return self.personal_data
    
    def build_user_profile(self) -> UserProfile:
        """Build user profile from loaded data."""
        # Extract skills
        if not self.linkedin_data.skills.empty:
            self.user_profile.skills = self.linkedin_data.skills["skill"].tolist()
        
        # Extract positions
        if not self.linkedin_data.positions.empty:
            for _, row in self.linkedin_data.positions.iterrows():
                self.user_profile.positions.append({
                    "title": row.get("title", ""),
                    "company": row.get("company_name", ""),
                    "description": row.get("description", ""),
                    "start_date": row.get("started_on", ""),
                    "end_date": row.get("finished_on", "")
                })
        
        # Extract profile info
        if not self.linkedin_data.profile.empty:
            profile_row = self.linkedin_data.profile.iloc[0]
            self.user_profile.name = profile_row.get("first_name", "") + " " + profile_row.get("last_name", "")
            self.user_profile.headline = profile_row.get("headline", "")
            self.user_profile.summary = profile_row.get("summary", "")
        
        # Extract preferences
        if not self.personal_data.preferences.empty:
            for _, row in self.personal_data.preferences.iterrows():
                category = row.get("category", "").lower()
                value = row.get("value", "")
                weight = float(row.get("weight", 1.0))
                
                if category == "sector":
                    self.user_profile.target_sectors.append(value)
                elif category == "location":
                    self.user_profile.target_locations.append(value)
                elif category == "contract_type":
                    self.user_profile.target_contract_types.append(value)
                
                self.user_profile.preferences[f"{category}:{value}"] = weight
        
        return self.user_profile
    
    def _load_csv(
        self,
        filepath: Path,
        processors: Optional[List[callable]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load a CSV file with optional processing."""
        try:
            # Try different encodings
            encodings = ["utf-8", "latin-1", "cp1252"]
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                logger.error(f"Could not load {filepath} with any encoding")
                return pd.DataFrame()
            
            # Apply processors
            if processors:
                for processor in processors:
                    df = processor(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return pd.DataFrame()
    
    def _process_connections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process connections data."""
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Add derived fields
        if "company" in df.columns:
            df["company_normalized"] = df["company"].apply(self._normalize_company_name)
            df["sector"] = df["company"].apply(self._infer_sector)
        
        if "position" in df.columns:
            df["seniority"] = df["position"].apply(self._infer_seniority)
        
        return df
    
    def _process_skills(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process skills data."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Rename if needed
        if "name" in df.columns and "skill" not in df.columns:
            df = df.rename(columns={"name": "skill"})
        
        # Normalize skills
        if "skill" in df.columns:
            df["skill_normalized"] = df["skill"].str.lower().str.strip()
        
        return df
    
    def _process_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process positions/experience data."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Parse dates
        date_columns = ["started_on", "finished_on", "start_date", "end_date"]
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        return df
    
    def _process_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process messages data."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Parse dates
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        return df
    
    def _process_target_companies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process target companies data."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Ensure required columns
        required = ["company_name", "sector", "priority"]
        for col in required:
            if col not in df.columns:
                df[col] = ""
        
        # Normalize
        df["company_normalized"] = df["company_name"].apply(self._normalize_company_name)
        df["sector_normalized"] = df["sector"].apply(self._normalize_sector)
        df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(3)
        
        return df
    
    def _process_job_offers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process job offers data."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Parse dates
        if "date_added" in df.columns:
            df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        
        # Create combined text for embedding
        text_cols = ["title", "company", "description", "requirements"]
        df["combined_text"] = df.apply(
            lambda row: " ".join(str(row.get(col, "")) for col in text_cols if col in df.columns),
            axis=1
        )
        
        return df
    
    def _process_preferences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process preferences data."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Ensure required columns
        if "weight" not in df.columns:
            df["weight"] = 1.0
        else:
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
        
        return df
    
    def _process_contacts_notes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process contact notes data."""
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        # Parse dates
        if "last_contact" in df.columns:
            df["last_contact"] = pd.to_datetime(df["last_contact"], errors="coerce")
        
        # Ensure priority
        if "priority" not in df.columns:
            df["priority"] = 3
        else:
            df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(3)
        
        return df
    
    def _normalize_company_name(self, name: str) -> str:
        """Normalize company name for matching."""
        if pd.isna(name):
            return ""
        
        name = str(name).lower().strip()
        
        # Remove common suffixes
        suffixes = [
            r"\s+(inc\.?|ltd\.?|llc|corp\.?|sa|sas|sarl|gmbh|ag|plc)$",
            r"\s+(france|uk|usa|europe)$"
        ]
        for suffix in suffixes:
            name = re.sub(suffix, "", name, flags=re.IGNORECASE)
        
        return name.strip()
    
    def _normalize_sector(self, sector: str) -> str:
        """Normalize sector name."""
        if pd.isna(sector):
            return ""
        
        sector_lower = str(sector).lower().strip()
        
        for normalized, variants in SECTOR_MAPPINGS.items():
            if any(v in sector_lower for v in variants):
                return normalized
        
        return sector_lower
    
    def _infer_sector(self, company: str) -> str:
        """Infer sector from company name."""
        if pd.isna(company):
            return ""
        
        company_lower = str(company).lower()
        
        # Known company mappings
        company_sectors = {
            "canal+": "media",
            "tf1": "media",
            "nike": "sports",
            "adidas": "sports",
            "bnp": "banking",
            "société générale": "banking",
            "axa": "insurance",
            "google": "tech",
            "microsoft": "tech",
            "amazon": "tech",
        }
        
        for key, sector in company_sectors.items():
            if key in company_lower:
                return sector
        
        return ""
    
    def _infer_seniority(self, title: str) -> str:
        """Infer seniority level from job title."""
        if pd.isna(title):
            return "unknown"
        
        title_lower = str(title).lower()
        
        seniority_keywords = {
            "c-level": ["ceo", "cto", "cfo", "coo", "chief"],
            "director": ["director", "directeur", "directrice", "vp", "vice president"],
            "manager": ["manager", "responsable", "head of", "lead"],
            "senior": ["senior", "sr.", "principal", "expert"],
            "junior": ["junior", "jr.", "stagiaire", "intern", "apprenti"],
            "mid": ["analyst", "engineer", "consultant", "specialist"]
        }
        
        for level, keywords in seniority_keywords.items():
            if any(kw in title_lower for kw in keywords):
                return level
        
        return "mid"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of loaded data."""
        return {
            "linkedin": {
                "connections": len(self.linkedin_data.connections),
                "skills": len(self.linkedin_data.skills),
                "positions": len(self.linkedin_data.positions),
                "messages": len(self.linkedin_data.messages)
            },
            "personal": {
                "target_companies": len(self.personal_data.target_companies),
                "job_offers": len(self.personal_data.job_offers),
                "preferences": len(self.personal_data.preferences),
                "contacts_notes": len(self.personal_data.contacts_notes)
            },
            "profile": {
                "name": self.user_profile.name,
                "skills_count": len(self.user_profile.skills),
                "positions_count": len(self.user_profile.positions),
                "target_sectors": self.user_profile.target_sectors,
                "target_locations": self.user_profile.target_locations
            }
        }
