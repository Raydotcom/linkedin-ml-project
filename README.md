# 🔗 LinkedIn Smart Recommender - Rayan HOBBALLAH, Nathan GEHIN, Kevin KONAN

An intelligent **LinkedIn recommendation system** built with **Streamlit**, designed to analyze LinkedIn data and deliver **actionable, ranked recommendations** for jobs, contacts, companies, and content using **NLP-based scoring**.

This project is designed for **practical career strategy**, not vanity dashboards.

---

## 🚀 Features

### 🎯 Job Recommendations
- Personalized job suggestions based on:
  - Semantic similarity (NLP embeddings)
  - Skills matching
  - Sector relevance
  - Location compatibility
  - Network proximity
- Adjustable **minimum score threshold**
- **Detailed score breakdown** per job
- Direct links to job offers
- Sorting by score, date, or company

---

### 👥 Contact Recommendations
- Identification of **high-value contacts** in your LinkedIn network
- Ranking based on relevance score
- Seniority filtering:
  - C-Level
  - Director
  - Manager
  - Senior
  - Mid
  - Junior
- Notes and annotations per contact
- One-click access to LinkedIn profiles
- Contact action buttons (profile / contact)

---

### 🏢 Target Company Identification
- Detection of companies strategically aligned with your profile
- Estimation of:
  - Number of contacts inside the company
  - Number of job openings
- Priority-based visual indicators
- Ability to follow companies directly
- Company cards with score and network density

---

### 📰 Content Recommendations
- Suggestion of relevant LinkedIn content based on:
  - Your skills
  - Your professional interests
- NLP-powered relevance scoring
- Direct access to external content

---

### 📈 Analytics Dashboard
- Network size overview
- Target companies tracking
- Contacts inside target companies
- Saved job offers
- Top skills ranking
- Sector distribution of your network

---

## 🧠 Architecture Overview

```text
.
├── app.py                      # Main Streamlit application
├── src/
│   ├── recommender.py          # Core recommendation engine
│   ├── config.py               # Global configuration
│   ├── utils.py                # Formatting & helper utilities
│   └── data_loader.py          # LinkedIn & personal data ingestion
├── data/
│   ├── linkedin/
│   │   └── Connections.csv     # Exported LinkedIn connections
│   └── personal/               # User-defined targets & notes
├── requirements.txt
└── README.md
```

# LinkedIn Smart Recommender

## ⚙️ Technologies Used
* **Python 3.9+**
* **Streamlit** – interactive web interface
* **Pandas** – data manipulation
* **Plotly** – interactive charts
* **Sentence Transformers** – semantic similarity (NLP embeddings)
* **Logging** – application monitoring and debugging

## 🖥️ UI & UX Highlights
* **Dark-mode compatible**
* **Custom CSS:**
    * Metric cards
    * Recommendation cards
    * Score badges
* **Color-coded scoring system**
* **Responsive multi-column layout**
* **Expandable score breakdowns**
* **Cached model and data loading for performance**

## 📊 Scoring Logic
Each recommendation receives a final score between **0 and 1**.

### 🔢 Scoring Dimensions
| Criterion | Description |
| :--- | :--- |
| **Semantic** | NLP similarity between your profile and the item |
| **Skills** | Overlap between required and owned skills |
| **Sector** | Industry alignment |
| **Location** | Geographic compatibility |
| **Network** | Shared connections and proximity |

The final score is a weighted aggregation of these dimensions.

### 🟢🟡🔴 Score Interpretation
Scores are displayed as percentages and classified into three levels:

| Level | Range | Visual Badge |
| :--- | :--- | :--- |
| 🟢 **High** | $\ge 70\%$ | Green background |
| 🟡 **Medium** | $\ge 40\%$ and $< 70\%$ | Yellow background |
| 🔴 **Low** | $< 40\%$ | Red background |

## 🧩 Configuration
Main configuration is controlled via `src/config.py`.

### Minimum Score Threshold
`config.recommendation.min_score_threshold`

This parameter:
* Defines the default minimum score shown in recommendations
* Is adjustable live from the sidebar slider
* Filters out low-relevance items

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone [https://github.com/yourusername/linkedin-smart-recommender.git](https://github.com/yourusername/linkedin-smart-recommender.git)
cd linkedin-smart-recommender
```
### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Add LinkedIn data
Optional (but recommended):
- Personal target companies
- Saved job offers
- Contact notes
- Preferred sectors and locations

## 🛠️ Known Limitations

Recommendation quality depends heavily on data richness

Content recommendations require a dedicated content dataset

Network analysis limited to exported LinkedIn data

No official LinkedIn API integration (manual export required)

## ❤️ Credits
Built with Streamlit and Sentence Transformers.

Designed for data-driven career decisions, not guesswork.

## 📄 License
This project is provided for educational and personal use.
