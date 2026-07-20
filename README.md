# Telecommunications Customer Churn Prediction

## Project Overview

This repository contains a machine-learning system that predicts whether a telecommunications customer is likely to churn.

## Dataset

The project uses the Telco Customer Churn dataset containing customer demographic, contract, service and billing information.

## Models Evaluated

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Final Model

Logistic Regression was selected based on cross-validation performance and its low generalisation gap.

## Final Test Performance

- Accuracy: 0.8055
- Precision: 0.6572
- Recall: 0.5588
- F1-score: 0.6040
- ROC-AUC: 0.8420

## Repository Contents

- `COM763_Telco_Churn_Prediction.ipynb`
- `app.py`
- `telco_churn_model.pkl`
- `requirements.txt`
- `Telco-Customer-Churn.csv`

## Application

The Streamlit application accepts customer information and estimates customer churn probability.
