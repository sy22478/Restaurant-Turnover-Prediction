# Restaurant-Turnover-Prediction
Hackathon Model Ensemble Project

This project details the development of a weighted ensemble model combining LightGBM (LGBM), XGBoost, and CatBoost for a hackathon. The goal was to achieve a low Root Mean Squared Error (RMSE) on the competition's evaluation metric.

Overview

The project involved:

Feature Engineering: Processing and creating a significant number of features (34) to train the individual models.

Hyperparameter Optimization: Utilizing Optuna to find optimal hyperparameters for each of the three base models (LGBM, XGBoost, CatBoost).

Ensemble Creation: Combining the predictions of the tuned base models using a weighted averaging approach.

Evaluation: Assessing the performance of the ensemble model based on the RMSE metric, achieving a final score of 12.2M RMSE.

Technologies Used

Python

LightGBM

XGBoost

CatBoost

Optuna

Project Structure

data/: Contains the training and testing datasets.

notebooks/: Includes Jupyter notebooks for data exploration, feature engineering, model training, and ensemble creation.

scripts/: Might contain Python scripts for modularized tasks.

models/: Could store the trained model files.

requirements.txt: Lists the project dependencies.

Getting Started

Clone the repository:
git clone

Install dependencies:
pip install -r requirements.txt

Place data: Ensure the training and testing data are placed in the data/ directory (or modify the data loading paths in the notebooks/scripts).

Run the notebooks/scripts: Execute the Jupyter notebooks in the notebooks/ directory sequentially to reproduce the model training and ensemble creation process.

Further Work

Experiment with different weighting strategies for the ensemble.

Explore additional feature engineering techniques.

Consider incorporating other high-performing models into the ensemble.

Analyze the individual model predictions to gain insights into their strengths and weaknesses.

Contact

Sonu Yadav
sonu.yadav19997@gmail.com
