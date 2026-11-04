<div align="center">

# 🎓 Student Performance Prediction

### Predicting students’ final Mathematics grades with machine learning

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Regression-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Google Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

**A regression project exploring how students’ demographic, social, and academic characteristics relate to final Mathematics grades.**

</div>

---

## 📌 Project at a glance

| | |
|---|---|
| **Task** | Supervised regression |
| **Target** | Final grade (`G3`), scored from 0 to 20 |
| **Dataset** | UCI Student Performance — Mathematics |
| **Observations** | 395 |
| **Features** | 33 |
| **Best reported model** | Random Forest Regressor · **R² 0.8530** |

> **What’s interesting?** The full-feature model performs substantially better than the early-prediction experiment, which excludes the previous-period grades `G1` and `G2`.

## 🧭 Contents

- [Project goals](#-project-goals)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Models](#-models)
- [Results](#-results)
- [Run the notebook](#-run-the-notebook)
- [Limitations](#-limitations)
- [Repository structure](#-repository-structure)
- [Conclusion](#-conclusion)

## 🎯 Project goals

- Explore student academic performance through exploratory data analysis (EDA).
- Identify features associated with final Mathematics grades.
- Apply feature-selection and preprocessing techniques.
- Train and compare regression models using appropriate evaluation metrics.
- Investigate early prediction without `G1` and `G2`.

## 📊 Dataset

The project uses the **Student Performance** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance), focusing on the Mathematics subject.

| Attribute | Description |
|:--|:--|
| **Records** | 395 students |
| **Features** | 33 |
| **Target** | `G3` — final Mathematics grade (0–20) |
| **Prediction type** | Regression |

## 🔬 Workflow

1. Load and inspect the data
2. Check data quality and clean as needed
3. Perform exploratory data analysis
4. Examine correlations and multicollinearity
5. Select features and preprocess data
6. Split data into training and test sets
7. Scale features where appropriate
8. Develop regression models
9. Optimize model hyperparameters
10. Compare model performance
11. Analyze residuals and prediction errors
12. Assess model stability and learning curves

## 🤖 Models

The notebook implements and compares:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Hyperparameter-tuned Random Forest
- Early-prediction Random Forest (**excluding `G1` and `G2`**)

## 📈 Results

The following scores are reported for the test dataset. **Lower MAE and RMSE are better; higher R² is better.**

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|:--|--:|--:|--:|
| Linear Regression | 1.3482 | 2.1222 | 0.7804 |
| Ridge Regression | 1.3474 | 2.1214 | 0.7805 |
| Lasso Regression | 1.2688 | 2.0828 | 0.7884 |
| **Random Forest** | **1.0482** | **1.7359** | **0.8530** |
| Tuned Random Forest | 1.0696 | 1.7417 | 0.8521 |
| Early Prediction Random Forest | 3.3248 | 4.0275 | 0.2089 |

### 🏆 Best reported result: Random Forest

The original Random Forest Regressor achieved the strongest reported test-set results:

- **MAE:** 1.0482
- **RMSE:** 1.7359
- **R²:** 0.8530

The early-prediction model’s lower score highlights how much more challenging prediction becomes when prior-period grades (`G1` and `G2`) are unavailable.

## ▶️ Run the notebook

1. Open [`Student_Performance_Prediction_ML.ipynb`](Student_Performance_Prediction_ML.ipynb) in this repository.
2. Open it in [Google Colab](https://colab.research.google.com/) or run it in a local Jupyter environment.
3. Obtain the Mathematics dataset from the [UCI dataset page](https://archive.ics.uci.edu/dataset/320/student+performance).
4. Upload or place the dataset where the notebook’s data-loading cell expects it.
5. Run the notebook cells in order.

**Libraries used:** Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

## ⚠️ Limitations

- The dataset contains only **395 observations**, so results may not generalize to broader student populations.
- Previous-period grades are strongly informative for final-grade prediction.
- The early-prediction model has considerably lower reported performance.
- Further validation on independent, additional student data is needed before any real-world use.

## 📁 Repository structure

```text
Student-Performance-Prediction/
├── Student_Performance_Prediction_ML.ipynb
└── README.md
```

## ✅ Conclusion

This project demonstrates an end-to-end machine-learning workflow for predicting student academic performance. Among the models evaluated, **Random Forest Regression** achieved the strongest reported test-set performance. The early-prediction comparison also illustrates the importance of prior-period grades and the need to interpret predictive performance in context.

---

<div align="center">

**Project category:** Machine Learning · Data Science · Regression  
**Environment:** Google Colab

</div>
