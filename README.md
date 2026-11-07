# CreditGuard – Intelligent Credit Risk Assessment System

CreditGuard is a machine learning-based credit risk assessment application that predicts whether a loan applicant belongs to the good-credit or bad-credit category based on historical financial and demographic information.

## Project Overview

Financial institutions need to assess the creditworthiness of applicants before approving loans. CreditGuard uses machine learning classification models to support this assessment by identifying applicants who may represent higher credit risk.

## Features

- Credit risk classification using machine learning.
- Exploratory Data Analysis (EDA).
- Data preprocessing and feature engineering.
- Logistic Regression and Random Forest models.
- Model evaluation using accuracy, precision, recall, F1-score and ROC-AUC.
- Interactive Streamlit dashboard.
- Applicant-level predictions.

## Dataset

**Dataset:** UCI Statlog (German Credit Data)

- 1,000 applicant records.
- 20 original input attributes.
- 700 good-credit records.
- 300 bad-credit records.

Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Machine Learning Workflow

1. Dataset collection and inspection.
2. Data cleaning and preprocessing.
3. Exploratory Data Analysis.
4. Categorical feature encoding.
5. Numerical feature scaling.
6. Train-test split.
7. Model training.
8. Model evaluation and comparison.
9. Feature importance analysis.
10. Deployment using Streamlit.

## Installation and Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CreditGuard_Project
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

http://localhost:8501

## Project Structure

```text
CreditGuard_Project/
│
├── app.py
├── creditguard_artifacts/
│   ├── creditguard_model.pkl
│   └── metadata.json
├── requirements.txt
├── README.md
└── .gitignore
```

## Important Note

The model predicts historical good-credit or bad-credit categories using the German Credit dataset. Its predicted class probability should not be interpreted as a calibrated probability of future loan default.

This project is intended for educational and analytical purposes and should not be used as the sole basis for real-world lending decisions.

## Author

Shriyansh Goyal  
B.Tech – Electronics and Communication Engineering  
Netaji Subhas University of Technology (NSUT), Delhi