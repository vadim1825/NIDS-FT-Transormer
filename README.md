# FT-Transformer for Network Intrusion Detection (NIDS)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Overview

This project implements a FT-Transformer (Feature Tokenizer Transformer) architecture to detect network intrusions using the UNSW-NB15 dataset.

Unlike traditional machine learning models (e.g., Random Forest, XGBoost) often used for tabular data, this project leverages Deep Learning and Self-Attention mechanisms. It adapts the Transformer architecture (famous in NLP) to process heterogeneous tabular features, allowing the model to learn complex interactions between network traffic attributes.

## 🚀 Key Features

- **Deep Learning on Tabular Data**: Implements the paper "Revisiting Deep Learning Models for Tabular Data" using TensorFlow/Keras.
- **Custom Keras Layers**: Features a custom `AddCLSToken` layer that mimics BERT's aggregation mechanism, allowing the model to gather global context into a single learnable vector.
- **Robust Preprocessing**:
    - SMOTE (Synthetic Minority Over-sampling Technique) to handle severe class imbalance in attack types.
    - Feature Tokenizer: Embeds categorical variables and projects numerical variables into a unified latent space.
- **Experiment Pipeline**: Automated training loop with 3 independent runs to ensure statistical significance of the results.

## 🧠 Model Architecture

The model follows this data flow:

1. **Input**: Heterogeneous data (numerical & categorical).
2. **Feature Tokenizer**:
    - Categorical features → Embedding Lookup.
    - Numerical features → Dense Layer + LayerNorm.
3. **[CLS] Token**: A learnable token is appended to the beginning of the sequence (implemented via custom layer).
4. **Transformer Encoder**: *N* blocks of Multi-Head Self-Attention and Feed-Forward Networks.
5. **Classification Head**: Extracts the [CLS] token vector and passes it through a dense layer for final Softmax prediction.

## 📂 Project Structure
```text
├── results/
│   ├── images/      # Generated training plots (Loss/Accuracy)
│   ├── logs/        # TensorBoard logs
│   └── models/      # Saved .keras models
├── UNSW_NB15_training_set_csv.csv  # Dataset file
├── main.py          # Main execution script
├── requirements.txt # Dependencies
└── README.md        # Project documentation
```

## 📊 Results & Visualization

The training pipeline generates metrics plots automatically. Below is an example of the training performance across 3 runs:

- **Average Accuracy**: ~XX.XX% (Update this after running)
- **F1-Score**: ~0.XX (Update this after running)

## 🛠️ Installation & Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Dataset**:
    - Download the `UNSW_NB15_training_set_csv.csv`.
    - Place it in the root directory (or update `CONFIG["PATH"]` in `main.py`).

4. **Run the training pipeline**:
    ```bash
    python main.py
    ```

## 📦 Requirements

- tensorflow
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib