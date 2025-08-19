# AI’s Industrial Impacts and Policy Coverage (2014–2025): A News-Based Sentiment Analysis

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 📌 Overview
This research project analyzes over 22,000 news articles from 2014 to 2025 to investigate the evolving media discourse on Artificial Intelligence (AI). The study focuses on its impact on specific job roles, associated public sentiment, and prevalent policy recommendations using Natural Language Processing (NLP) and sentiment analysis techniques.

## 🎯 Key Features
- **Large-Scale Analysis:** Processes a corpus of 22,000+ news articles.
- **Advanced NLP:** Utilizes a fine-tuned transformer model for accurate sentiment and entity recognition.
- **Role-Centric Insights:** Categorizes job roles by perceived AI impact (Augmented, At Risk, Obsolete, Emerging).
- **Policy Analysis:** Extracts and analyzes discussed policy recommendations like upskilling and AI ethics governance.
- **Interactive Visualizations:** Includes generated charts and graphs for clear result interpretation.

## 📂 Repository Structure




sentiment_analysis_project/

│
├── 📊 charts/ # Generated visualizations (e.g., .png, .jpg)

├── ⚙️ preprocessing.py # Data cleaning, tokenization, and preprocessing

├── 📈 sentiment_analysis_visualise.py # Analysis and visualization logic

├── 📄 report.md # Research report with methodology and results

├── 🔐 .gitignore # Git exclusion rules (e.g., data, secrets)

└── 📖 README.md # Project documentation and setup guide

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. Then install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn transformers requests
