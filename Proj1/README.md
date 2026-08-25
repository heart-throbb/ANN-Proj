# Proj1 — ANN (Artificial Neural Network)

This folder contains a small project demonstrating a feed-forward neural network pipeline using Keras/TensorFlow. It includes notebooks for experiments and prediction, a minimal Flask app for serving the trained model, and pre-processing artifacts required for inference.

## Contents

- 1-Experiments.ipynb — Notebook with data exploration, model architecture experiments, training, and evaluation.
- 2-Prediction.ipynb / prediction.ipynb — Notebook(s) showing the prediction/inference pipeline.
- 3-app.py — Minimal Flask app to serve the trained model for predictions.
- requirements.txt — Python package dependencies required to run the notebooks/app.
- model.h5 — Trained Keras model (binary file).
- scaler.pkl — Scaler used for numeric feature normalization.
- label_encoder_gender.pkl — Label encoder used for gender column.
- onehot_encoder_geography.pkl — One-hot encoder used for geography column.
- Dataset/ — Folder containing the dataset used for training.
- logs/ — Training logs (e.g., TensorBoard). Note: logs/fit may contain large or ephemeral files.

## Requirements

- Python 3.8+
- Install dependencies:

```
pip install -r Proj1/requirements.txt
```

## Quick start

1. Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

2. Install dependencies:

```
pip install -r Proj1/requirements.txt
```

3. Run experiments:

Open `Proj1/1-Experiments.ipynb` in Jupyter/VSCode and run the cells to reproduce training.

4. Make predictions:

Use `Proj1/2-Prediction.ipynb` or `Proj1/prediction.ipynb` to run the prediction pipeline with sample inputs.

5. Run the app:

```
python Proj1/3-app.py
```

The app exposes a prediction endpoint — see `3-app.py` for usage.

## Notes about artifacts

- `model.h5`, `scaler.pkl`, and encoder files are required for inference. If you plan to re-train, these files may be overwritten.
- Consider storing large artifacts outside the Git repo (e.g., S3 or GitHub Releases) to keep the repository lightweight.

## Logs management

Training logs under `Proj1/logs/fit` can grow large. Recommended actions:

- Remove tracked log files from the repository and add `Proj1/logs/` to `.gitignore` to prevent re-adding them.
- Or store logs externally.

To remove tracked log files locally and update the remote, run:

```
git rm -r --cached Proj1/logs/fit
git commit -m "Remove Proj1/logs/fit from repository"
git push origin <branch>
```

(Replace `<branch>` with the branch you use, e.g., `main`.)

## Author

Repository owner: heart-throbb

If you need me to also remove the tracked log files from the repository (delete them from the Git history), tell me and I will provide or perform the steps.