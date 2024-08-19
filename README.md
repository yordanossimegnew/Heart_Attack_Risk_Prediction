# Heart Attack Risk Prediction

![heart attack](https://github.com/yordanossimegnew/End_to_End_Heart_Attack_Risk_Prediction/blob/main/heart%20attack.jpg)

![web app gif](https://github.com/yordanossimegnew/Heart_Attack_Risk_Prediction/blob/main/real%20time%20app.gif)

![confusion matrix](https://github.com/yordanossimegnew/End_to_End_Heart_Attack_Risk_Prediction/blob/main/reports/figure/confusion_matrix.jpg)

This repository contains an end-to-end machine learning project aimed at predicting the risk of heart attacks using various patient health metrics. The project utilizes Python and various libraries for data processing, feature engineering, model training, and web deployment.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model](#model)
- [Web Application](#web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview

Heart disease is a leading cause of mortality worldwide. This project aims to predict the risk of heart attacks based on patient data, helping in early diagnosis and prevention. The machine learning pipeline includes data preprocessing, feature engineering, model training, and evaluation. A FastAPI web application is also developed to allow users to input data and get predictions.

## Project Structure

```plaintext
.
├── data
│   ├── processed_data
│   │   └── selected_features.csv
│   └── raw_data
│       └── heart.csv
├── models
│   └── lrmodel.joblib
│   └── scaler.joblib
├── notebooks
│   └── 1.Data_Analysis.ipynb
│   └── 2.Feature_Engineering.ipynb
│   └── 3.Feature_Selection.ipynb
│   └── 4.Model_Building.ipynb
├── reports
│   └── figure
│       └── confusion_matrix.jpg
│       └── correlation_analysis.jpg
│       └── feature_importances.jpg
├── scripts
│   ├── data
│   │   └── processed_data
│   │       └── selected_features.csv
│   │       └── x_test.csv
│   │       └── x_train.csv
│   │       └── y_test.csv
│   │       └── y_train.csv
│   ├── raw_data
│   │   └── heart.csv
│   ├── templates
│   │   └── index.html
│   ├── config.py
│   ├── main.py
│   ├── pipeline.py
│   ├── preprocess.py
├── requirements.txt
├── README.md
```

## Data

The dataset used in this project is a heart disease dataset that includes various health metrics such as age, sex, chest pain type, resting blood pressure, cholesterol level, and more. The raw data is stored in `data/raw_data/heart.csv`.

## Model

The model used for prediction is a Logistic Regression model. The pipeline involves:
- **Reciprocal transformation** for certain features
- **Yeo-Johnson transformation** for others
- **Standard Scaling** for all features

## Web Application

A FastAPI web application is developed to allow users to input their health metrics and get predictions on their risk of heart attack. The web app uses a form to collect user input and displays the prediction result.

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Analysis and Model Training:**
   Open the notebooks in the `notebooks` directory to explore data analysis, feature engineering, and model training steps.

2. **Running the Web Application:**
   Navigate to the `scripts` directory and run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

3. **Access the Web Application:**
   Open your browser and go to `http://127.0.0.1:8000` to access the application.

## Results

The model is evaluated using several metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Confusion matrix and feature importance plots are provided in the `reports/figure` directory.
