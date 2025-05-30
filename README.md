# 🎧 Music & Mental Health: Predictive Analysis of Emotional Outcomes

**Author:** \[Arsen Jamkotchian]
**Project Type:** Data Science Portfolio
**Directory Structure:** [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/)

## 📘 Project Overview

This project explores the relationship between music listening habits and mental health outcomes using the [Music & Mental Health Survey Dataset](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results). Specifically, the goal is to **predict whether listening to music led to an improvement, worsening, or no effect on an individual's mental health**.

The project covers the full data science lifecycle:

* Data cleaning and preprocessing
* Exploratory data analysis (EDA)
* Feature engineering
* Model training and evaluation
* Interpretation and communication of results

## 📂 Repository Structure

This repository follows the [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/) standard:

```
├── data
│   ├── external        # Raw external data (e.g., original CSVs)
│   ├── interim         # Intermediate data that has been transformed
│   ├── processed       # Cleaned data ready for modeling
│   └── raw             # Original, immutable data dump
│
├── docs                # Documentation and report files
│
├── models              # Trained and serialized models, model predictions
│
├── notebooks           # Jupyter notebooks for exploration and modeling
│
├── references          # Data dictionaries, references, and external literature
│
├── reports
│   └── figures         # Generated graphics and plots
│
├── src
│   ├── data            # Data loading, cleaning, and transformation scripts
│   ├── features        # Feature engineering scripts
│   ├── models          # Training, prediction, and evaluation scripts
│   └── visualization   # Scripts for generating plots and EDA
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## 🧠 Problem Statement

Can we predict the impact of music listening on an individual's mental health?
Target variable:

* **mental\_health\_effect** (categorical):

  * `Improved`
  * `No effect`
  * `Worsened`

This is a **multi-class classification problem**.

## 📊 Dataset

**Source:** [Music & Mental Health Survey Dataset on Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)

**Key Features:**

* Demographic information (age, country, etc.)
* Music listening habits (genres, hours per day, whether they listen while studying/working)
* Mental health indicators and perceived effects of music

**Target Variable (engineered):**

* Derived from survey responses regarding whether music has improved, worsened, or had no effect on mental health.

## 🔍 Project Goals

* Clean and preprocess messy survey data
* Understand relationships between listening habits and mental health outcomes
* Train and evaluate classification models (e.g., Random Forest, XGBoost, Logistic Regression)
* Communicate findings visually and narratively

## 🚧 Project Status

* 🔲 Data loading and cleaning
* 🔲 Exploratory data analysis
* 🔲 Feature engineering
* 🔲 Model training and evaluation
* 🔲 Reporting and visualization
* 🔲 Final portfolio writeup

## 🛠️ Technologies Used

* Python (Pandas, scikit-learn, XGBoost, Matplotlib, Seaborn)
* Jupyter Notebooks
* cookiecutter-data-science project structure
* Kaggle dataset integration

## 📈 Example Results (To Be Added)

*This section will include charts and metrics once the model is finalized.*

## 📌 Future Work

* Experiment with NLP on open-ended responses
* Explore genre-specific effects in more depth
* Create a web dashboard for interactive data exploration

## 📄 License

This project is for educational and portfolio purposes. Dataset copyright belongs to the original creators.
