# 🍽️ Restaurant-Turnover-Prediction
## Hackathon Model Ensemble Project

A machine learning hackathon project for predicting restaurant annual turnover in India using ensemble methods (LightGBM, XGBoost, CatBoost). This project demonstrates advanced ensemble techniques, hyperparameter optimization with Optuna, and comprehensive feature engineering to achieve a final score of **12.2M RMSE** on the competition leaderboard.

---

## 📊 Overview

The project involved:

- **Feature Engineering**: Processing and creating a significant number of features (34) from restaurant data including location, opening dates, cuisine types, social media metrics, Zomato ratings, customer surveys, and mystery visitor data
- **Hyperparameter Optimization**: Utilizing Optuna to find optimal hyperparameters for each of the three base models (LightGBM, XGBoost, CatBoost) with custom objective functions
- **Ensemble Creation**: Combining the predictions of the tuned base models using a weighted averaging approach on log-transformed predictions
- **Evaluation**: Assessing the performance of the ensemble model based on the RMSE metric, achieving a final score of **12.2M RMSE**

### Business Context
**Domain:** Restaurant industry in India with diverse culinary neighborhoods

**Data Sources:**
- Restaurant details (location, opening dates, cuisine types)
- Social media metrics (Facebook/Instagram Popularity Quotient)
- Zomato ratings and customer surveys
- Mystery visitor data

**Problem:** Predict Annual Turnover based on restaurant characteristics and market factors to help stakeholders make data-driven decisions about restaurant investments, expansion strategies, and operational improvements.

## Complete Architecture

### ML Pipeline Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │    │  Feature         │    │   Model         │
│   Processing    │───►│  Engineering     │───►│   Training      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  Data           │             │
         │              │  Preprocessing  │             │
         │              └─────────────────┘             │
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  Hyperparameter │             │
         │              │  Optimization   │             │
         │              │  (Optuna)       │             │
         │              └─────────────────┘             │
         │                                               │
         └──────────────────────┬──────────────────────┘
                                │
                    ┌──────────▼──────────┐
                    │  Ensemble Model     │
                    │  (LightGBM+XGBoost+ │
                    │   CatBoost)         │
                    └─────────────────────┘
```

## Complete Tech Stack & ML Implementation

### Advanced Machine Learning Architecture
```python
# Actual Ensemble Implementation (from hackathon_1.py)
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
import optuna

# Weighted Ensemble with Log-Scale Predictions
ensemble_test_predictions_log = (normalized_weight_lgbm * test_preds_lgbm +
                                normalized_weight_xgb * test_preds_xgb +
                                normalized_weight_cat * test_preds_cat)

# Convert back to original scale
ensemble_test_predictions_orig = np.expm1(ensemble_test_predictions_log)
ensemble_test_predictions_orig[ensemble_test_predictions_orig < 0] = 0

# Hyperparameter Optimization with Optuna
def objective_lgbm(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    # K-Fold cross-validation with RMSE evaluation
```

### Machine Learning Implementation Details (Verified)
- **Algorithms:** LightGBM, XGBoost, CatBoost regressors
- **Ensemble Method:** Weighted averaging on log-transformed predictions
- **Hyperparameter Optimization:** Optuna for each model with custom objective functions
- **Cross-Validation:** K-Fold cross-validation with RMSE scoring
- **Target Transformation:** Log transformation with np.log1p() and np.expm1() for scaling
- **Target Metric:** RMSE optimization with 12.2M threshold

### Data Processing & Feature Engineering (Actual Implementation)
- **Date Processing:** Opening Day of Restaurant converted to Restaurant_Age_Days, Opening_Year, Opening_Month, Opening_DayOfWeek
- **Numerical Imputation:** Facebook/Instagram Popularity Quotient with median imputation
- **Categorical Handling:** Various restaurant attributes and location data
- **Scaling:** StandardScaler for numerical features
- **Data Combination:** Train/test concatenation for consistent feature engineering

### Development Environment
- **Platform:** Google Colab with drive.mount() integration
- **Package Installation:** pip install optuna catboost xgboost scikit-learn category_encoders
- **Data Source:** Google Drive CSV files (Train_dataset.csv, Test_dataset.csv)
- **Output:** submission_ensemble.csv, submission_tuned.csv files

### Business Context (From Code Documentation)
- **Domain:** Restaurant industry in India with diverse culinary neighborhoods
- **Data Sources:** Restaurant details, social media metrics, Zomato ratings, customer surveys, mystery visitor data
- **Problem:** Predict Annual Turnover based on restaurant characteristics and market factors

## Skills Developed

### Advanced Machine Learning (Verified Implementation)
- **Ensemble Techniques:** Log-scale weighted averaging of LightGBM, XGBoost, CatBoost predictions
- **Hyperparameter Optimization:** Optuna trials with custom objective functions for each algorithm
- **Feature Engineering:** Date transformation, social media metrics processing, categorical encoding
- **Model Selection:** Comprehensive 3-model ensemble with normalized weight optimization

### Data Science Methodology
- **Exploratory Data Analysis:** Comprehensive data understanding, pattern identification
- **Feature Creation:** Date-time features, categorical encoding, numerical transformations
- **Model Validation:** Cross-validation strategies, overfitting prevention
- **Performance Optimization:** RMSE minimization, model interpretability

### Business Analytics
- **Restaurant Domain Knowledge:** Understanding business metrics, revenue drivers
- **Predictive Modeling:** Turnover forecasting, business impact analysis
- **Competition Strategy:** Hackathon optimization, leaderboard performance

## Technical Achievements (Verified Implementation)
- **Target Performance:** RMSE below 12.2M threshold optimization
- **Code Structure:** 527 lines of production-quality Python code
- **Model Ensemble:** 3-algorithm weighted averaging (LightGBM + XGBoost + CatBoost)
- **Feature Engineering:** Date processing (4 new features), social media metrics, categorical handling
- **Optimization:** Optuna hyperparameter tuning for each model with custom objectives
- **Data Pipeline:** Train/test concatenation, consistent preprocessing, log-scale transformations
- **Output Generation:** Automated submission file creation (submission_ensemble.csv)

---

## 🛠️ Technologies Used

- **Python**: Primary programming language for data analysis and ML implementation
- **LightGBM**: Gradient boosting framework for high-performance models
- **XGBoost**: Extreme gradient boosting for robust predictions
- **CatBoost**: Gradient boosting library with categorical feature support
- **Optuna**: Hyperparameter optimization framework with custom objective functions
- **scikit-learn**: Machine learning utilities (StandardScaler, cross-validation)
- **pandas**: Data manipulation and preprocessing
- **numpy**: Numerical computing and array operations
- **category_encoders**: Categorical feature encoding

---

## 🗂️ Project Structure

```
Restaurant-Turnover-Prediction/
├── data/                           # Training and testing datasets
│   ├── Train_dataset.csv           # Training data with Annual Turnover
│   └── Test_dataset.csv            # Test data for predictions
├── notebooks/
│   └── hackathon_1.ipynb           # Main Jupyter notebook (527 lines)
├── scripts/
│   └── hackathon_1.py              # Python script version
├── models/                         # Trained model files
│   ├── lgbm_model.pkl
│   ├── xgboost_model.pkl
│   └── catboost_model.pkl
├── submissions/
│   ├── submission_ensemble.csv     # Final ensemble predictions
│   └── submission_tuned.csv        # Tuned model predictions
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sy22478/Restaurant-Turnover-Prediction.git
cd Restaurant-Turnover-Prediction
```

### 2. Install Dependencies
```bash
pip install optuna catboost xgboost scikit-learn category_encoders pandas numpy jupyter
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 3. Place Data
Ensure the training and testing data are placed in the `data/` directory (or modify the data loading paths in the notebooks/scripts):
- `data/Train_dataset.csv`
- `data/Test_dataset.csv`

### 4. Run the Notebooks/Scripts
Execute the Jupyter notebook to reproduce the model training and ensemble creation process:

**Option A: Google Colab (Recommended)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Upload Train_dataset.csv and Test_dataset.csv to Google Drive
# Open hackathon_1.ipynb in Colab
# Run all cells sequentially
```

**Option B: Local Jupyter**
```bash
jupyter notebook notebooks/hackathon_1.ipynb
```

---

## 📈 Model Performance

### Final Ensemble Results
- **RMSE Score**: **12.2M** (below target threshold)
- **Ensemble Method**: Weighted averaging on log-transformed predictions
- **Base Models**: LightGBM, XGBoost, CatBoost

### Ensemble Implementation
```python
# Weighted Ensemble with Log-Scale Predictions
ensemble_test_predictions_log = (normalized_weight_lgbm * test_preds_lgbm +
                                normalized_weight_xgb * test_preds_xgb +
                                normalized_weight_cat * test_preds_cat)

# Convert back to original scale
ensemble_test_predictions_orig = np.expm1(ensemble_test_predictions_log)
ensemble_test_predictions_orig[ensemble_test_predictions_orig < 0] = 0
```

### Hyperparameter Optimization (Optuna)
```python
def objective_lgbm(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    # K-Fold cross-validation with RMSE evaluation
```

---

## 🔧 Feature Engineering

### Date Processing
Created 4 new features from Opening Day of Restaurant:
- `Restaurant_Age_Days`: Days since restaurant opening
- `Opening_Year`: Year of restaurant opening
- `Opening_Month`: Month of restaurant opening
- `Opening_DayOfWeek`: Day of week when restaurant opened

### Numerical Imputation
- **Facebook/Instagram Popularity Quotient**: Median imputation for missing social media metrics

### Categorical Handling
- Various restaurant attributes (cuisine type, location)
- Location data encoding

### Scaling
- **StandardScaler**: Applied to numerical features for consistent model training

### Data Combination
- Train/test concatenation for consistent feature engineering across datasets

---

## 💡 Further Work

- **Experiment with different weighting strategies** for the ensemble (optimize weights using validation set)
- **Explore additional feature engineering techniques**:
  - Interaction features (e.g., cuisine × location)
  - Time-based features (seasonality, trends)
  - Social media engagement ratios
- **Consider incorporating other high-performing models** into the ensemble (e.g., Random Forest, Neural Networks)
- **Analyze the individual model predictions** to gain insights into their strengths and weaknesses
- **Feature selection**: Use SHAP values or permutation importance to identify most critical features
- **Stacking ensemble**: Implement meta-learner on top of base models
- **Target encoding**: Explore advanced categorical encoding techniques

---

## 📋 Requirements (requirements.txt)

```
pandas
numpy
scikit-learn
category_encoders
lightgbm
xgboost
catboost
optuna
jupyter
matplotlib
seaborn
```

---

## 📧 Contact

For questions, improvements, or collaboration:

- **Email**: sonu.yadav19997@gmail.com
- **LinkedIn**: [Sonu Yadav](https://www.linkedin.com/in/sonu-yadav-a61046245/)
- **GitHub**: [@sy22478](https://github.com/sy22478)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project demonstrates advanced ensemble techniques, hyperparameter optimization, and comprehensive feature engineering for competitive machine learning hackathons, achieving strong RMSE performance through weighted model averaging.*
