# Project 4: Credit Card Fraud Detection using ANN

This project uses a feed-forward Artificial Neural Network (ANN) to detect potentially fraudulent credit card transactions.

It focuses on binary classification with highly imbalanced transaction data and demonstrates preprocessing, feature scaling, class weighting, ANN model development, evaluation, threshold analysis, and Streamlit deployment.

## Problem Statement

The goal is to classify transactions into two categories:

```text
0 -> Normal Transaction
1 -> Fraudulent Transaction
```

Fraud detection is challenging because fraudulent transactions represent a very small portion of the overall dataset.

## Project Structure

```text
Proj4/
├── 1-FraudDetection.ipynb
├── 2-Prediction.ipynb
├── 3-app.py
├── README.md
├── Dataset/
│   └── creditcard.csv
├── model.keras
├── scaler.pkl
└── logs/
    └── fit/
```

## Dataset

The project uses the Credit Card Fraud Detection dataset.

The dataset contains the following columns:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class`

The `Class` column is the target variable:

```text
0 -> Normal Transaction
1 -> Fraudulent Transaction
```

### Important Dataset Characteristic

The dataset is highly imbalanced. Normal transactions significantly outnumber fraudulent transactions.

Because of this imbalance, accuracy alone is not enough to evaluate the model. The project also uses:

- Precision
- Recall
- F1 score
- ROC-AUC
- Precision-Recall AUC
- Confusion matrix

## Preprocessing

The following preprocessing steps are performed:

1. Load the dataset.
2. Remove duplicate rows.
3. Separate features and target.
4. Split the dataset into training and testing sets.
5. Use stratification to preserve the class distribution.
6. Standardize features using `StandardScaler`.

The scaler is fitted only on the training data and then used to transform both the training and testing data.

The fitted scaler is saved as:

```text
scaler.pkl
```

## ANN Architecture

The project uses a feed-forward Artificial Neural Network:

```text
Input
  ↓
Dense(128, ReLU)
  ↓
Dropout(0.3)
  ↓
Dense(64, ReLU)
  ↓
Dropout(0.3)
  ↓
Dense(32, ReLU)
  ↓
Dense(1, Sigmoid)
```

No CNN, RNN, LSTM, GRU, or Transformer layers are used.

## Model Compilation

The model uses:

- Optimizer: Adam
- Loss: Binary crossentropy
- Metrics:
  - Accuracy
  - Precision
  - Recall

## Handling Class Imbalance

Because fraudulent transactions are rare, class weights are calculated using:

```python
compute_class_weight(class_weight="balanced")
```

The resulting class weights are supplied to the ANN during training. This helps the model pay more attention to the minority fraud class.

## Training

The model uses:

- Epochs: 30
- Batch size: 256
- Validation split: 20%
- Early stopping
- TensorBoard
- Class weights

Early stopping restores the best model weights based on validation loss.

## Threshold Analysis

The default classification threshold is:

```text
0.5
```

Fraud detection often requires adjusting the threshold depending on the priority:

- Detecting more fraud
- Reducing false alarms
- Balancing precision and recall

The training notebook evaluates multiple thresholds.

## Model Files

The trained ANN is saved as:

```text
model.keras
```

The preprocessing scaler is saved as:

```text
scaler.pkl
```

Both files are required for inference.

## Prediction Notebook

`2-Prediction.ipynb` demonstrates how to:

- Load the trained ANN.
- Load the saved scaler.
- Load transaction data.
- Apply the same preprocessing used during training.
- Generate fraud probabilities.
- Convert probabilities into fraud or normal predictions.

## Streamlit Application

The project includes a Streamlit application:

```text
3-app.py
```

Run it from the `Proj4` directory:

```bash
python -m streamlit run 3-app.py
```

The application allows users to enter transaction feature values and receive:

- Fraud probability
- Normal or fraud prediction
- Prediction result

## TensorBoard

Training logs are stored under:

```text
logs/fit/
```

Start TensorBoard from the `Proj4` directory:

```bash
tensorboard --logdir logs/fit
```

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib streamlit tensorboard
```

## Run the Project

### Step 1: Add the Dataset

Place the dataset at:

```text
Proj4/Dataset/creditcard.csv
```

### Step 2: Train the Model

Open:

```text
1-FraudDetection.ipynb
```

Run all cells from top to bottom. This creates:

```text
model.keras
scaler.pkl
logs/
```

### Step 3: Test Inference

Open:

```text
2-Prediction.ipynb
```

This notebook loads the saved model and scaler without retraining.

### Step 4: Run the Streamlit App

From the `Proj4` directory, run:

```bash
python -m streamlit run 3-app.py
```

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- TensorBoard

## Why This Project Is Useful

This project expands the ANN portfolio from basic classification and regression into a real-world imbalanced classification problem.

```text
                    ANN
                     |
       --------------+--------------
       |             |             |
       v             v             v
    PROJ 1        PROJ 2        PROJ 3
 Classification  Regression     Images
       |             |             |
       v             v             v
   Binary          Numeric       Multi-class
       |             |             |
       --------------+--------------
                     |
                     v
                  PROJ 4
                     |
                     v
             Fraud Detection
                     |
          -----------+-----------
          |          |          |
          v          v          v
       Imbalance   Recall   Thresholds
          |          |          |
          -----------+-----------
                     |
                     v
              ANN + Class Weights
```

The project remains purely ANN-based:

```text
Input
  |
Dense
  |
Dropout
  |
Dense
  |
Dropout
  |
Dense
  |
Sigmoid
  |
Prediction
```

No recurrent or convolutional architecture is involved.

## Author

Repository owner: `heart-throbb`
