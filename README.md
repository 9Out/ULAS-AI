# 🚀 ULAS-AI: Dynamic Google Play Store App Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)

ULAS-AI is a practical, research-oriented sentiment analysis dashboard for Google Play Store reviews. It combines live review scraping, multilingual/informal text normalization, machine learning classification, and interactive visual analytics into a single Streamlit application. 

The system accepts a Google Play Store application URL or app ID, extracts user reviews, normalizes noisy Indonesian text, and classifies each review as positive or negative. Ultimately, this project serves as a usable example of applying NLP techniques to app-user feedback analysis in an Indonesian-language context.

## 📌 Project Overview

The application helps evaluate user perception of an application by analyzing review text from Google Play Store. It is designed for:
- Identifying sentiment polarity in user reviews.
- Summarizing dominant complaints and positive feedback.
- Comparing negative review patterns by app version.
- Visualizing confidence and review distribution.
- Supporting quick product feedback analysis for app teams.

The user interface is implemented in Streamlit and the actual analysis workflow is driven by a saved scikit-learn pipeline stored in the `model/` directory.

## 🎯 Background and Objectives

The project is focused on sentiment analysis for Indonesian-language app reviews. In practice, Play Store reviews often include slang, abbreviations, misspellings, repetitive punctuation, and mixed expressions. The preprocessing stage in `preprocess.py` is designed to normalize that noisy language before classification.

The main objective is to transform raw user review data into an interpretable sentiment report that can help app owners understand public perception, identify recurring issues, and monitor sentiment shifts across versions.

## ✨ Key Features

- **App Metadata Lookup:** Fetch details via Google Play Store URL or app ID.
- **Dynamic Scraping:** Real-time review scraping using `google-play-scraper`.
- **Text Preprocessing:** Robust Indonesian text normalization for slang and noise removal.
- **Sentiment Classification:** High-accuracy prediction using a saved ML pipeline.
- **Interactive Dashboard:** Sentiment distribution and confidence analysis visualization.
- **Version Analysis:** Identify specific complaints tied to application versions.

## 🛠️ Technology Stack

| Category | Technologies / Libraries |
| --- | --- |
| **Language** | Python 3.12+ |
| **Frontend / Dashboard** | Streamlit |
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn, joblib |
| **NLP Preprocessing** | NLTK, regex, unicodedata |
| **Visualization** | Plotly, Matplotlib, WordCloud |
| **App Review Scraping** | google-play-scraper |

## 📁 Project Structure

```text
sentiment_app/
├── .devcontainer/               # VS Code / Codespaces container configuration
├── data/
│   └── dataset_playstore_labeled4.csv
├── model/
│   ├── playstore_sentiment_pipeline_v10.joblib
│   ├── playstore_sentiment_pipeline_v8.joblib
│   └── playstore_sentiment_pipeline_v4.joblib
├── tampilan_web/
│   ├── DistribusiSentimen_TingkatKeyakinan.jpg
│   ├── GrafikKeluhan_Wordcloud.jpg
│   ├── Informasi_aplikasi.jpg
│   ├── KataDominan.jpg
│   ├── Ulasan_representatif.jpg
│   └── home.jpg
├── .gitignore                   # Ignored files (e.g., env/, __pycache__/)
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
├── app.py                       # Main Streamlit application entry point
├── preprocess.py                # Text cleaning and slang normalization logic
└── requirements.txt             # Python package dependencies
```

## System Architecture

The application architecture is relatively simple and follows a direct data-processing flow:

```mermaid
flowchart LR
    A[User input: Play Store URL or App ID] --> B[get_app_info() in app.py]
    B --> C[google_play_scraper.app() / reviews()]
    C --> D[DataFrame of review text + metadata]
    D --> E[preprocess_playstore()]
    E --> F[Loaded ML pipeline from model/*.joblib]
    F --> G[Sentiment prediction: Positif / Negatif]
    G --> H[Plotly charts, confidence metrics, complaint summaries]
    H --> I[Streamlit dashboard]
```


## Prerequisites
Required software:

- Python 3.12+
- pip
- Git (optional, for version control)
- A browser to open the Streamlit dashboard

## Installation Guide

### 1. Clone or open the project

```bash
git clone https://github.com/9Out/ULAS-AI.git
cd ULAS-AI
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Linux / macOS:

```
python -m venv env
source env/bin/activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## How to Run the Application

From the project root directory, run the following command to start the Streamlit dashboard:

```powershell
streamlit run app.py
```

## AI / ML Implementation

The trained model is stored in the joblib file `model/playstore_sentiment_pipeline_v10.joblib`.

Verified runtime metadata from the model payload:

- `version`: `v10.0.0`
- `model`: `Logistic Regression`
- `feature_extraction`: `FeatureUnion(word_tfidf + char_tfidf)`
- `labels`: `{0: 'Negatif', 1: 'Positif'}`
- `accuracy`: `0.977`
- `confusion_matrix`: `[[1627, 24], [22, 327]]`
- `metrics.cv_f1_mean`: `0.8946986055563291`

### Text preprocessing approach

The logic in `preprocess.py` tackles informal Indonesian shorthand (e.g., gak, bgt, tdk) through Unicode normalization, lowercase conversion, and dictionary-based slang replacement to ensure high prediction accuracy.

## Screenshots / Application Preview

The project includes several dashboard screenshots in `tampilan_web/`:

- [tampilan_web/home.jpg](tampilan_web/home.jpg)
- [tampilan_web/Informasi_aplikasi.jpg](tampilan_web/Informasi_aplikasi.jpg)
- [tampilan_web/DistribusiSentimen_TingkatKeyakinan.jpg](tampilan_web/DistribusiSentimen_TingkatKeyakinan.jpg)
- [tampilan_web/KataDominan.jpg](tampilan_web/KataDominan.jpg)
- [tampilan_web/GrafikKeluhan_Wordcloud.jpg](tampilan_web/GrafikKeluhan_Wordcloud.jpg)
- [tampilan_web/Ulasan_representatif.jpg](tampilan_web/Ulasan_representatif.jpg)


## License

This project is distributed under the MIT License. The full license text is available in `LICENSE`.

## Summary

ULAS-AI is a practical, research-oriented sentiment analysis dashboard for Google Play Store reviews. It combines live review scraping, multilingual/informal text normalization, machine learning classification, and interactive visual analytics into a single Streamlit application. The project is a usable example of applying NLP and classification techniques to app-user feedback analysis in an Indonesian-language context.
