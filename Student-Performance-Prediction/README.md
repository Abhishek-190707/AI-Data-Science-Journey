Student Performance Prediction Using Machine Learning

Project Overview

This project focuses on predicting students' final Mathematics grades using supervised machine learning regression algorithms.

The project uses the Student Performance dataset from the UCI Machine Learning Repository. It explores the relationship between students' demographic, social and academic characteristics and their final academic performance.

The main objective is to develop, evaluate and compare different regression models for predicting students' final grades.

Project Objective

- Analyse student academic performance through Exploratory Data Analysis (EDA).
- Identify important features associated with final Mathematics grades.
- Apply feature selection and preprocessing techniques.
- Build and compare multiple regression models.
- Evaluate model performance using appropriate regression metrics.
- Explore early prediction without using previous-period grades.

Dataset Information

Attribute| Description
Dataset| Student Performance
Source| UCI Machine Learning Repository
Subject| Mathematics
Total Records| 395
Total Features| 33
Target Variable| G3
Prediction Type| Regression

The target variable "G3" represents the student's final Mathematics grade, ranging from 0 to 20.

Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

Project Workflow

1. Data Loading and Initial Inspection
2. Data Cleaning and Quality Checks
3. Exploratory Data Analysis
4. Correlation and Multicollinearity Analysis
5. Feature Selection
6. Train-Test Split
7. Feature Scaling
8. Regression Model Development
9. Hyperparameter Optimization
10. Model Evaluation and Comparison
11. Residual and Error Analysis
12. Model Stability Analysis
13. Learning Curve Analysis

Machine Learning Models

The following regression models were implemented:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Hyperparameter-Tuned Random Forest
6. Early Prediction Random Forest (excluding G1 and G2)

Model Performance

The following table summarises the results obtained on the test dataset.

Model| MAE| RMSE| R² Score
Linear Regression| 1.3482| 2.1222| 0.7804
Ridge Regression| 1.3474| 2.1214| 0.7805
Lasso Regression| 1.2688| 2.0828| 0.7884
Random Forest| 1.0482| 1.7359| 0.8530
Tuned Random Forest| 1.0696| 1.7417| 0.8521
Early Prediction Random Forest| 3.3248| 4.0275| 0.2089

Best Performing Model

The original Random Forest Regressor achieved the best overall performance, with:

- MAE: 1.0482
- RMSE: 1.7359
- R² Score: 0.8530

The early prediction experiment showed that predicting final grades without previous-period grades ("G1" and "G2") is considerably more challenging.

Repository Structure

Student-Performance-Prediction/
│
├── Student_Performance_Prediction_ML.ipynb
└── README.md

How to Run the Project

1. Download the notebook from this repository.
2. Open Google Colab.
3. Upload the ".ipynb" notebook.
4. Obtain the Student Performance Mathematics dataset from the UCI Machine Learning Repository.
5. Upload or provide the dataset at the location expected by the notebook.
6. Execute the notebook cells sequentially.

Limitations

- The dataset contains only 395 observations.
- Previous-period grades have a strong influence on final grade predictions.
- The early prediction model has considerably lower predictive performance.
- The models require validation on additional student data before real-world application.

Conclusion

This project demonstrates an end-to-end machine learning workflow for predicting student academic performance.

Among the evaluated models, Random Forest Regression achieved the strongest predictive performance. The project also highlights the importance of feature selection, model evaluation, hyperparameter optimization and understanding model limitations.

---

Project Category: Machine Learning | Data Science | Regression

Environment: Google Colab
