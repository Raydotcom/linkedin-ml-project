"""
LinkedIn Smart Recommender - Streamlit Application
===================================================

Main application interface for the LinkedIn recommendation system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.recommender import LinkedInRecommender
from src.config import config
from src.utils import format_score, get_color_for_score, truncate_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LinkedIn Smart Recommender",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0077B5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #0077B5;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .score-high { background: #dcfce7; color: #166534; }
    .score-medium { background: #fef9c3; color: #854d0e; }
    .score-low { background: #fee2e2; color: #991b1b; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_recommender():
    """Initialize and cache the recommender."""
    recommender = LinkedInRecommender()
    recommender.load_data()
    return recommender


def render_header():
    """Render the application header."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<p class="main-header">🔗 LinkedIn Smart Recommender</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Recommandations intelligentes basées sur votre profil et vos préférences</p>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Rafraîchir les données", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()


def render_sidebar(recommender: LinkedInRecommender):
    """Render the sidebar with stats and filters."""
    with st.sidebar:
        st.markdown("## 📊 Tableau de bord")
        
        # Data summary
        summary = recommender.data_loader.get_summary()
        
        st.markdown("### Données LinkedIn")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Contacts", summary["linkedin"]["connections"])
            st.metric("Compétences", summary["linkedin"]["skills"])
        with col2:
            st.metric("Expériences", summary["linkedin"]["positions"])
            st.metric("Messages", summary["linkedin"]["messages"])
        
        st.markdown("### Données Personnelles")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Entreprises cibles", summary["personal"]["target_companies"])
            st.metric("Offres sauvegardées", summary["personal"]["job_offers"])
        with col2:
            st.metric("Préférences", summary["personal"]["preferences"])
            st.metric("Notes contacts", summary["personal"]["contacts_notes"])
        
        st.markdown("---")
        
        # Profile summary
        st.markdown("### 👤 Votre Profil")
        profile = summary["profile"]
        if profile["name"]:
            st.write(f"**Nom:** {profile['name']}")
        st.write(f"**Compétences:** {profile['skills_count']}")
        st.write(f"**Expériences:** {profile['positions_count']}")
        
        if profile["target_sectors"]:
            st.write(f"**Secteurs cibles:** {', '.join(profile['target_sectors'])}")
        
        if profile["target_locations"]:
            st.write(f"**Localisations:** {', '.join(profile['target_locations'])}")
        
        st.markdown("---")
        
        # Filters
        st.markdown("### ⚙️ Paramètres")
        min_score = st.slider(
            "Score minimum",
            min_value=0.0,
            max_value=1.0,
            value=config.recommendation.min_score_threshold,
            step=0.05,
            help="Afficher uniquement les recommandations au-dessus de ce score"
        )
        
        return {"min_score": min_score}


def render_score_badge(score: float) -> str:
    """Render a score badge with appropriate styling."""
    percentage = score * 100
    if score >= 0.7:
        css_class = "score-high"
    elif score >= 0.4:
        css_class = "score-medium"
    else:
        css_class = "score-low"
    
    return f'<span class="score-badge {css_class}">{percentage:.0f}%</span>'


def render_score_breakdown(score_breakdown: dict):
    """Render a score breakdown chart."""
    labels = {
        "semantic": "Similarité",
        "skills": "Compétences",
        "sector": "Secteur",
        "location": "Localisation",
        "network": "Réseau"
    }
    
    data = []
    for key, value in score_breakdown.items():
        if key in labels:
            data.append({
                "Critère": labels[key],
                "Score": value * 100
            })
    
    if not data:
        return
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x="Score",
        y="Critère",
        orientation="h",
        color="Score",
        color_continuous_scale=["#ef4444", "#eab308", "#22c55e"],
        range_color=[0, 100]
    )
    
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        coloraxis_showscale=False,
        xaxis_title="",
        yaxis_title=""
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_job_recommendations(recommender: LinkedInRecommender, filters: dict):
    """Render job recommendations tab."""
    st.markdown("## 🎯 Offres d'emploi recommandées")
    
    # Search and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Rechercher",
            placeholder="Ex: Data Analyst Python Paris...",
            key="job_search"
        )
    
    with col2:
        top_k = st.selectbox(
            "Nombre de résultats",
            options=[5, 10, 20, 50],
            index=1,
            key="job_top_k"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Trier par",
            options=["Score", "Date", "Entreprise"],
            key="job_sort"
        )
    
    # Get recommendations
    if search_query:
        results = recommender.search(search_query, search_type="jobs", top_k=top_k)
    else:
        results = recommender.recommend_jobs(top_k=top_k, min_score=filters["min_score"])
    
    # Display results
    if not results.recommendations:
        st.info("💡 Aucune offre d'emploi trouvée. Ajoutez des offres dans `data/personal/job_offers.csv`")
        
        with st.expander("📝 Format du fichier job_offers.csv"):
            st.code("""title,company,description,location,url,date_added,status
Data Analyst Stage,CANAL+,"Analyse de données sportives, Python, SQL",Paris,https://...,2024-01-15,interested
Data Scientist,Nike,"Machine learning pour recommandations",Paris,https://...,2024-01-10,applied""")
        return
    
    st.markdown(f"**{results.total_count} recommandations** (temps de calcul: {results.query_time:.2f}s)")
    
    for rec in results.recommendations:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"### {rec.title}")
                st.markdown(f"**{rec.subtitle}**")
                
                if rec.details.get("description"):
                    st.markdown(truncate_text(rec.details["description"], 300))
                
                # Tags
                tags = []
                if rec.details.get("skills", {}).get("matched"):
                    tags.extend([f"✅ {s}" for s in rec.details["skills"]["matched"][:3]])
                if rec.details.get("sector", {}).get("match"):
                    tags.append(f"🏢 {rec.details['sector'].get('job_sector', '')}")
                if rec.details.get("network", {}).get("connections", 0) > 0:
                    tags.append(f"👥 {rec.details['network']['connections']} contacts")
                
                if tags:
                    st.markdown(" • ".join(tags))
            
            with col2:
                st.markdown(render_score_badge(rec.score), unsafe_allow_html=True)
                
                if rec.url:
                    st.link_button("Voir l'offre →", rec.url, use_container_width=True)
            
            with st.expander("📊 Détails du score"):
                render_score_breakdown(rec.score_breakdown)
            
            st.markdown("---")


def render_contact_recommendations(recommender: LinkedInRecommender, filters: dict):
    """Render contact recommendations tab."""
    st.markdown("## 👥 Contacts recommandés")
    
    # Search and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Rechercher un contact",
            placeholder="Ex: Recruteur Data CANAL+...",
            key="contact_search"
        )
    
    with col2:
        top_k = st.selectbox(
            "Nombre de résultats",
            options=[10, 20, 50, 100],
            index=1,
            key="contact_top_k"
        )
    
    with col3:
        seniority_filter = st.multiselect(
            "Niveau",
            options=["C-Level", "Director", "Manager", "Senior", "Mid", "Junior"],
            default=[],
            key="contact_seniority"
        )
    
    # Get recommendations
    if search_query:
        results = recommender.search(search_query, search_type="contacts", top_k=top_k)
    else:
        results = recommender.recommend_contacts(top_k=top_k, min_score=filters["min_score"])
    
    # Display results
    if not results.recommendations:
        st.info("💡 Aucun contact trouvé. Exportez vos contacts LinkedIn dans `data/linkedin/Connections.csv`")
        return
    
    st.markdown(f"**{results.total_count} contacts recommandés**")
    
    # Grid layout
    cols = st.columns(2)
    
    for i, rec in enumerate(results.recommendations):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{rec.title}</h4>
                    <p>{rec.subtitle}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    # Seniority badge
                    seniority = rec.details.get("seniority", {}).get("level", "")
                    if seniority:
                        st.caption(f"📊 {seniority.title()}")
                
                with col2:
                    st.markdown(render_score_badge(rec.score), unsafe_allow_html=True)
                
                # Notes if available
                if rec.details.get("notes"):
                    st.info(f"📝 {rec.details['notes']}")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if rec.url:
                        st.link_button("Voir le profil", rec.url, use_container_width=True)
                with col2:
                    if st.button("📧 Contacter", key=f"contact_{rec.id}", use_container_width=True):
                        st.session_state[f"draft_{rec.id}"] = True
                
                st.markdown("---")


def render_company_recommendations(recommender: LinkedInRecommender, filters: dict):
    """Render company recommendations tab."""
    st.markdown("## 🏢 Entreprises recommandées")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        top_k = st.selectbox(
            "Nombre de résultats",
            options=[10, 15, 25, 50],
            index=1,
            key="company_top_k"
        )
    
    with col2:
        include_from_network = st.checkbox(
            "Inclure les entreprises du réseau",
            value=False,
            key="company_include_network"
        )
    
    # Get recommendations
    results = recommender.recommend_companies(
        top_k=top_k,
        min_score=filters["min_score"],
        include_non_targets=include_from_network
    )
    
    if not results.recommendations:
        st.info("💡 Ajoutez vos entreprises cibles dans `data/personal/target_companies.csv`")
        
        with st.expander("📝 Format du fichier target_companies.csv"):
            st.code("""company_name,sector,priority,location,notes
CANAL+,Media/Sports,1,Paris,Diffusion sportive - data analytics
Nike,Sports/Retail,1,Paris,Sports analytics
BNP Paribas,Banking,2,Paris,Data science équipe risques""")
        return
    
    st.markdown(f"**{results.total_count} entreprises**")
    
    # Display as cards
    cols = st.columns(3)
    
    for i, rec in enumerate(results.recommendations):
        with cols[i % 3]:
            # Color based on priority
            priority = rec.details.get("priority", 3)
            border_color = "#22c55e" if priority == 1 else "#eab308" if priority == 2 else "#94a3b8"
            
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid {border_color};
                margin-bottom: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            ">
                <h4 style="margin: 0 0 0.5rem 0;">{rec.title}</h4>
                <p style="color: #666; margin: 0 0 0.5rem 0;">{rec.subtitle}</p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>👥 {rec.details.get('connections_count', 0)} contacts</span>
                    <span>💼 {rec.details.get('job_openings', 0)} offres</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(render_score_badge(rec.score), unsafe_allow_html=True)
            
            if rec.details.get("notes"):
                st.caption(f"📝 {rec.details['notes']}")


def render_content_recommendations(recommender: LinkedInRecommender, filters: dict):
    """Render content recommendations tab."""
    st.markdown("## 📰 Contenus recommandés")
    
    results = recommender.recommend_content()
    
    if not results.recommendations:
        st.info("💡 Cette fonctionnalité nécessite des données de contenus à analyser.")
        
        # Show suggested topics
        if results.metadata.get("suggested_topics"):
            st.markdown("### 🎯 Thèmes suggérés pour vous")
            
            topics = results.metadata["suggested_topics"]
            cols = st.columns(min(len(topics), 5))
            
            for i, topic in enumerate(topics[:10]):
                with cols[i % 5]:
                    st.button(f"#{topic}", key=f"topic_{i}", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💡 Comment ajouter des contenus")
        st.markdown("""
        Créez un fichier `data/personal/content_interests.csv` avec:
        ```csv
        topic,source,url,notes,priority
        Data Science,LinkedIn,https://...,Articles techniques,1
        Sports Analytics,Medium,https://...,Études de cas,2
        ```
        """)
        return
    
    # Display content cards
    for rec in results.recommendations:
        with st.container():
            st.markdown(f"### {rec.title}")
            st.markdown(rec.subtitle)
            
            col1, col2 = st.columns([4, 1])
            with col2:
                st.markdown(render_score_badge(rec.score), unsafe_allow_html=True)
            
            if rec.url:
                st.link_button("Lire →", rec.url)
            
            st.markdown("---")


def render_dashboard(recommender: LinkedInRecommender):
    """Render the analytics dashboard."""
    st.markdown("## 📈 Dashboard Analytics")
    
    dashboard_data = recommender.get_dashboard_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    network_stats = dashboard_data["network_stats"]
    
    with col1:
        st.metric(
            "Réseau total",
            network_stats["total_connections"],
            help="Nombre total de connexions LinkedIn"
        )
    
    with col2:
        st.metric(
            "Entreprises cibles",
            network_stats["target_companies"],
            help="Entreprises dans votre liste cible"
        )
    
    with col3:
        st.metric(
            "Contacts dans cibles",
            network_stats["connections_at_targets"],
            help="Connexions dans vos entreprises cibles"
        )
    
    with col4:
        st.metric(
            "Offres sauvegardées",
            network_stats["saved_jobs"],
            help="Offres d'emploi dans votre liste"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Compétences principales")
        
        top_skills = dashboard_data.get("top_skills", [])
        if top_skills:
            skills_df = pd.DataFrame({
                "Compétence": top_skills[:10],
                "Importance": range(len(top_skills[:10]), 0, -1)
            })
            
            fig = px.bar(
                skills_df,
                x="Importance",
                y="Compétence",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ajoutez vos compétences LinkedIn pour voir ce graphique")
    
    with col2:
        st.markdown("### 🏢 Distribution par secteur")
        
        sector_dist = dashboard_data.get("sector_distribution", {})
        if sector_dist:
            sector_df = pd.DataFrame({
                "Secteur": list(sector_dist.keys()),
                "Contacts": list(sector_dist.values())
            })
            
            fig = px.pie(
                sector_df,
                values="Contacts",
                names="Secteur",
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Les secteurs seront affichés une fois les données chargées")
    
    # Recommendations summary
    st.markdown("### 🎯 Résumé des opportunités")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        job_results = recommender.recommend_jobs(top_k=5)
        st.markdown("#### Top 5 Offres")
        for rec in job_results.recommendations[:5]:
            st.markdown(f"- **{rec.title}** ({format_score(rec.score)})")
    
    with col2:
        contact_results = recommender.recommend_contacts(top_k=5)
        st.markdown("#### Top 5 Contacts")
        for rec in contact_results.recommendations[:5]:
            st.markdown(f"- **{rec.title}** ({format_score(rec.score)})")
    
    with col3:
        company_results = recommender.recommend_companies(top_k=5)
        st.markdown("#### Top 5 Entreprises")
        for rec in company_results.recommendations[:5]:
            st.markdown(f"- **{rec.title}** ({format_score(rec.score)})")


def main():
    """Main application entry point."""
    render_header()
    
    # Check for data
    linkedin_path = config.paths.linkedin_data
    personal_path = config.paths.personal_data
    
    has_linkedin = any(linkedin_path.glob("*.csv"))
    has_personal = any(personal_path.glob("*.csv"))
    
    if not has_linkedin and not has_personal:
        st.warning("⚠️ Aucune donnée détectée. Veuillez ajouter vos fichiers CSV.")
        
        with st.expander("📚 Guide de démarrage", expanded=True):
            st.markdown("""
            ### 1. Exportez vos données LinkedIn
            
            1. Allez sur LinkedIn → **Paramètres** → **Confidentialité des données**
            2. Cliquez sur **Obtenir une copie de vos données**
            3. Téléchargez et extrayez les fichiers
            4. Copiez les fichiers CSV dans `data/linkedin/`
            
            ### 2. Créez vos fichiers personnels
            
            Créez ces fichiers dans `data/personal/`:
            
            - `target_companies.csv` - Vos entreprises cibles
            - `job_offers.csv` - Les offres qui vous intéressent
            - `preferences.csv` - Vos préférences (secteurs, lieux...)
            - `contacts_notes.csv` - Notes sur vos contacts
            
            ### 3. Relancez l'application
            
            ```bash
            streamlit run app.py
            ```
            """)
        return
    
    # Initialize recommender
    try:
        with st.spinner("Chargement des données et du modèle NLP..."):
            recommender = get_recommender()
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        logger.exception("Error loading recommender")
        return
    
    # Sidebar
    filters = render_sidebar(recommender)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Offres d'emploi",
        "👥 Contacts",
        "🏢 Entreprises",
        "📰 Contenus",
        "📈 Dashboard"
    ])
    
    with tab1:
        render_job_recommendations(recommender, filters)
    
    with tab2:
        render_contact_recommendations(recommender, filters)
    
    with tab3:
        render_company_recommendations(recommender, filters)
    
    with tab4:
        render_content_recommendations(recommender, filters)
    
    with tab5:
        render_dashboard(recommender)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<center>Built with ❤️ using Streamlit & Sentence Transformers</center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
