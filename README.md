# ANN-Proj

Welcome to ANN-Proj — a collection of Jupyter Notebooks and Python code exploring artificial neural networks (ANNs). This repository contains experiments, visualizations, and example implementations aimed at learning and demonstrating ANN concepts.

## What is implemented in both projects
Both Proj1 and Proj2 implement the following core features and components:

- Feed-forward Artificial Neural Networks (ANNs) built with Keras / TensorFlow.
- A preprocessing pipeline that includes:
  - Label encoding for categorical features (gender).
  - One-hot encoding for categorical features (geography / region).
  - Feature scaling/standardization using a Scaler (StandardScaler).
- Training notebooks that demonstrate data exploration, model architecture experiments, training loops, early stopping, and logging to TensorBoard.
- Saved model artifacts for inference (model.h5 / .keras files) and saved preprocessing artifacts (scaler.pkl, encoder .pkl files).
- Separate prediction notebooks demonstrating how to load preprocessing artifacts and the trained model to make predictions.
- A lightweight app to serve predictions:
  - Proj1 uses a minimal Flask app (Proj1/3-app.py).
  - Proj2 uses a Streamlit app (Proj2/3-app.py) for interactive inputs.
- Example scripts and instructions to run the notebooks and apps locally.

These shared components make it straightforward to reproduce training, evaluate models, and run inference locally or adapt the code for deployment.

## Quick start
1. Clone the repository:

   git clone https://github.com/heart-throbb/ANN-Proj.git
   cd ANN-Proj

2. Create and activate a Python environment (recommended):

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .\.venv\Scripts\activate  # Windows (PowerShell)

3. Install dependencies

- For Proj1:

   pip install -r Proj1/requirements.txt

- For Proj2 (if no requirements file):

   pip install pandas numpy scikit-learn tensorflow streamlit jupyter

4. Run the notebooks

- Start JupyterLab / Notebook and open the notebooks in the desired project folder:

   jupyter lab
   # or
   jupyter notebook

- Recommended order: run the training notebooks (Proj1/1-Experiments.ipynb or Proj2/1-SalaryRegression.ipynb) first to (re)train models, then run the prediction notebooks (Proj1/2-Prediction.ipynb, Proj1/prediction.ipynb, Proj2/2-Prediction.ipynb) to verify inference.

5. Run the apps

- Proj1 (Flask):

   python Proj1/3-app.py

- Proj2 (Streamlit):

   cd Proj2
   python -m streamlit run 3-app.py

## Project overview
- Proj1: A binary classification/regression ANN example with training notebooks, inference notebooks, a Flask prediction endpoint, and saved model + preprocessing artifacts. See `Proj1/README.md` for details.
- Proj2: A salary regression ANN built on the Churn Modelling dataset with Streamlit UI, training and prediction notebooks, and saved preprocessing & model artifacts. See `Proj2/README.md` for details.

## Notes about artifacts and repo size
- Model files and training logs can be large. Consider storing large artifacts (trained weights, logs) in external storage (S3, Google Drive, GitHub Releases) and keeping the repository lightweight.
- The preprocessing pickles and smaller model files are required for running the prediction notebooks and apps as-is.

## Contributing
Contributions, issues, and suggestions are welcome. If you implemented shared improvements (for example, consolidating preprocessing code or adding a unified API), mention them in a PR and update the relevant project README.

## License
If a LICENSE file is present, that file determines the repository license. If none is present and you want to add one, consider adding an OSI-approved license such as MIT, Apache-2.0, or GPL-3.0.

## Contact
For questions or help, open an issue on this repository.
