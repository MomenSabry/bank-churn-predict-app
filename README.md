# Bank Customer Churn Prediction

🚀 **Live Demo:**  
👉 https://momen-churn-prediction-h8bmbef6esfzfnhq.austriaeast-01.azurewebsites.net

A modern, user-friendly web application built with **Streamlit** that predicts customer churn for a banking institution using a production-grade **Random Forest Classifier**.

## Screenshots

<p align="center">
  <img src="Single Prediction.png" alt="Single Prediction Interface">
</p>

## Project Highlights

- Trained and evaluated six classification algorithms: Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Gradient Boosting  
- Addressed severe class imbalance (~80% stay / 20% churn) using **SMOTE** oversampling  
- Selected **Random Forest** as the final model due to its superior balance of accuracy, robustness, and interpretability on tabular data  
- Engineered a full end-to-end ML pipeline: data cleaning → categorical encoding → scaling → imbalance handling → modeling → model persistence  
- Developed an intuitive **Streamlit** interface featuring:  
  • Single-record prediction with churn probability and risk classification  
  • Batch prediction from CSV files with downloadable enriched results  
  • Input validation, clear visual feedback, and responsive layout  
- Prepared the project for seamless deployment on **Azure App Service** and **Streamlit Community Cloud**  
- Included professional documentation, structured folder layout, and continuous deployment readiness  

## Features

- Real-time single customer churn prediction with probability score  
- Bulk processing via CSV upload → automatic feature transformation → result download  
- Automatic one-hot encoding for `Geography` and `Gender`  
- Clean, modern UI with validation and informative messages  
- Model loaded efficiently using joblib  

## Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/MomenSabry/bank-churn-predict-app.git
cd bank-churn-predict-app

# 2. (Recommended) Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
