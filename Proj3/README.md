# Proj3 — Fashion MNIST Classification using ANN

This project uses an Artificial Neural Network to classify clothing items from the Fashion MNIST dataset. It includes a training notebook, prediction examples, and a Streamlit web application for interactive image classification.

## Project Contents

- `1-FashionClassification.ipynb`: Data loading, exploration, model training, evaluation, and prediction examples.
- `3-app.py`: Streamlit web application for uploading images and getting predictions.
- `model.keras`: Trained ANN model (TensorFlow format).
- `model.h5`: Trained ANN model (HDF5 format, for compatibility).
- `logs/`: TensorBoard training logs. This directory is excluded from Git.

## Dataset

**Fashion MNIST** contains 70,000 grayscale images (28×28 pixels) of 10 clothing categories:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

60,000 images are used for training and 10,000 for testing.

## Model Architecture

The ANN uses a fully connected feed-forward network:

```text
Input (28×28 image)
  → Flatten (784)
  → Dense (128, relu)
  → Dropout (0.2)
  → Dense (64, relu)
  → Dropout (0.2)
  → Dense (10, softmax)
```

**Total parameters**: 102,410

**Training details**:

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Early Stopping: Enabled to prevent overfitting
- Batch size: 32
- Validation split: 20%

**Performance**:

- Training accuracy: ~97%
- Testing accuracy: ~89%

## Installation

Create or activate a Python environment, then install the required packages:

```bash
python -m pip install tensorflow streamlit numpy pandas matplotlib pillow scikit-learn
```

**Required packages:**

- tensorflow
- streamlit
- numpy
- pandas
- matplotlib
- pillow
- scikit-learn

## Run the Streamlit App

From the `Proj3` directory:

```bash
python -m streamlit run 3-app.py
```

The app opens in your browser. Upload an image and click **Predict** to classify it. The app displays:

- Predicted class
- Confidence score
- Top 3 predictions with probabilities

**Tips**:

- Images are automatically resized to 28×28 and converted to grayscale.
- Best results with clear, centered clothing items.
- Simple backgrounds work better than complex ones.

## Run the Notebook

Open and run the training notebook:

```bash
jupyter notebook 1-FashionClassification.ipynb
```

This notebook includes:

- Data loading and visualization
- Class distribution analysis
- Model definition and training
- Evaluation metrics and confusion matrix
- Prediction examples with visualizations

## Notes

- The trained model files (`model.keras` and `model.h5`) are required for the Streamlit app.
- `logs/` is ignored in Git because TensorBoard logs are generated during training.
- Model predictions depend on the trained weights and data normalization.
- To retrain the model, run the notebook and update the model files.

## Author

Repository owner: heart-throbb
