# Project 2: Customer Salary Regression

This project uses an Artificial Neural Network (ANN) to predict a customer's estimated salary from the Churn Modelling dataset.

## Project Contents

- `1-SalaryRegression.ipynb`: Data preparation, model creation, training, and saving.
- `2-Prediction.ipynb`: Loads the trained model and predicts salary for sample customer data.
- `3-app.py`: Streamlit application for interactive salary predictions.
- `Dataset/Churn_Modelling.csv`: Input dataset.
- `model.h5`: Trained ANN model.
- `salary_regression_model.keras`: Keras model file.
- `gender_encoder.pkl`: Saved gender label encoder.
- `geography_encoder.pkl`: Saved geography one-hot encoder.
- `scaler.pkl`: Saved feature standardization scaler.
- `regressionlogs/`: TensorBoard training logs. This directory is excluded from Git.

## Model

The model predicts `EstimatedSalary` using these customer features:

- Credit score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of products
- Credit-card status
- Active-member status
- Exited status

The preprocessing pipeline label-encodes gender, one-hot-encodes geography, and standardizes the input features with `StandardScaler`.

The ANN architecture is:

```text
Input -> Dense(64, relu) -> Dense(32, relu) -> Dense(1)
```

The model is trained with the Adam optimizer and mean absolute error loss. Early stopping and TensorBoard callbacks are used during training.

## Installation

Create or activate a Python environment, then install the required packages:

```bash
python -m pip install pandas numpy scikit-learn tensorflow streamlit jupyter
```

## Run the Streamlit App

Run the command from this `Proj2` directory:

```bash
cd ANN/Proj2
python -m streamlit run 3-app.py
```

The app opens in a browser and provides input controls for making an estimated salary prediction.

## Run the Notebooks

Open the notebooks from the `Proj2` directory:

```bash
jupyter notebook
```

Run `1-SalaryRegression.ipynb` first if you want to retrain the model. Then run `2-Prediction.ipynb` to test a prediction.

## Notes

- The model and preprocessing files must remain in the same directory as `3-app.py`.
- `regressionlogs/` is ignored because TensorBoard logs are generated during training and are not required to run the application.
- Model predictions depend on the trained model weights and the saved preprocessing files.
