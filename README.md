# ANN-Proj

Learning projects built with Python, TensorFlow/Keras, scikit-learn, Jupyter notebooks, and Streamlit.

## Projects

| Project                  | Task                          | Application |
| ------------------------ | ----------------------------- | ----------- |
| [Proj1](Proj1/README.md) | Customer churn classification | Streamlit   |
| [Proj2](Proj2/README.md) | Estimated salary regression   | Streamlit   |

Both projects include data preprocessing, ANN training, saved model artifacts, prediction notebooks, and interactive applications. TensorBoard logs are generated during training and excluded from Git.

## Quick start

1. Clone the repository:

   ```bash
   git clone https://github.com/heart-throbb/ANN-Proj.git
   cd ANN-Proj
   ```

2. Create and activate a Python environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   python -m pip install -r Proj1/requirements.txt
   ```

   Proj2 uses the same dependencies. The requirements file in `Proj1` can be used for both projects.

4. Run the notebooks

- Start JupyterLab / Notebook and open the notebooks in the desired project folder:

  ```bash
  jupyter lab
  ```

  Run a training notebook first if you want to retrain a model. Then run its prediction notebook to verify inference.

5. Run the apps

   ```bash
   cd Proj1
   python -m streamlit run 3-app.py
   ```

   In a second terminal, run Proj2 from its directory:

   ```bash
   cd Proj2
   python -m streamlit run 3-app.py
   ```

## Project overview

- Proj1: A customer churn classification ANN built on the Churn Modelling dataset. See [Proj1/README.md](Proj1/README.md) for details.
- Proj2: An estimated salary regression ANN built on the Churn Modelling dataset. See [Proj2/README.md](Proj2/README.md) for details.

## Notes about artifacts and repo size

- Model files and preprocessing pickles are included so the prediction notebooks and apps can run as-is.
- Training logs are ignored by `.gitignore` because they are generated files.
- For larger production models, consider using external artifact storage or GitHub Releases.

## Contributing

Contributions, issues, and suggestions are welcome. If you implemented shared improvements (for example, consolidating preprocessing code or adding a unified API), mention them in a PR and update the relevant project README.

## License

No license file is currently included.

## Contact

For questions or help, open an issue on this repository.
