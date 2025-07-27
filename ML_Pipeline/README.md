# Telco Customer Churn Prediction - ML Pipeline

## 📌 Objective
Build an end-to-end machine learning pipeline to predict customer churn using the Telco Churn dataset. The goal is to identify customers likely to cancel services, enabling proactive retention strategies.

## 🛠️ Methodology
### Data Preparation
- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7043 customers, 20 features)
- **Cleaning**:
  - Dropped `customerID`
  - Converted `TotalCharges` to numeric
  - Handled missing values
- **Class Distribution**: 73.4% No Churn | 26.6% Churn

### Model Development
1. **Preprocessing**:
   - Numerical features: Standard Scaling
   - Categorical features: One-Hot Encoding
2. **Models**:
   - Logistic Regression
   - Random Forest Classifier
3. **Optimization**:
   - Hyperparameter tuning via `GridSearchCV`
   - Stratified 80/20 train-test split

## 📊 Key Results
### Best Parameters
| Model              | Optimal Parameters                          |
|--------------------|--------------------------------------------|
| Logistic Regression| `{'C': 10, 'solver': 'liblinear'}`         |
| Random Forest      | `{'max_depth': 5, 'n_estimators': 50}`     |

### Performance Metrics
| Model              | Accuracy | Precision (Churn) | Recall (Churn) |
|--------------------|----------|-------------------|----------------|
| Logistic Regression| 80%      | 0.65              | 0.55           |
| Random Forest      | 79%      | 0.65              | 0.47           |

### Confusion Matrices
**Logistic Regression**  