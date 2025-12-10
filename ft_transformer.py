import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

# Set non-interactive backend for saving plots without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers, regularizers, Model, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from typing import List, Dict, Tuple, Any

# --- Configuration Constants ---
# Using a centralized configuration dictionary allows for easy tuning
# without searching through the code ("Config-driven development").
CONFIG = {
    "NUM_CLASSES": 4,
    "EMBEDDING_DIM": 32,
    "NUM_HEADS": 2,
    "FFN_DIM": 64,
    "NUM_BLOCKS": 1,
    "DROPOUT": 0.1,
    "LR": 1e-3,
    "BATCH_SIZE": 256,
    "EPOCHS": 10,
    "L2_REG": 0.005,
    "PATH": 'UNSW_NB15_training_set_csv.csv',
    "DIRS": {
        'models': 'results/models',
        'logs': 'results/logs',
        'images': 'results/images'
    }
}

# Ensure output directories exist
for directory in CONFIG["DIRS"].values():
    os.makedirs(directory, exist_ok=True)


class AddCLSToken(layers.Layer):
    """
    Custom Keras Layer that appends a learnable [CLS] token to the beginning of the input sequence.

    This is a standard technique in Transformer models (like BERT) to aggregate
    global information from the entire sequence into a single vector for classification.

    Instead of using GlobalAveragePooling, we let the model learn how to aggregate
    information into this specific token via the Self-Attention mechanism.
    """

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        # Initialize the [CLS] token as a trainable weight
        self.cls_token = self.add_weight(
            shape=(1, 1, embedding_dim),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Get the dynamic batch size from the input tensor
        batch_size = tf.shape(inputs)[0]
        # Broadcast the token to match the batch size: (batch_size, 1, embedding_dim)
        cls_token = tf.tile(self.cls_token, [batch_size, 1, 1])
        # Concatenate along the sequence axis (axis=1) -> [CLS, Feature1, Feature2, ...]
        return tf.concat([cls_token, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'embedding_dim': self.embedding_dim})
        return config


def load_and_preprocess(filepath: str) -> Tuple[Tuple, Tuple, Tuple, List, List, Dict, List]:
    """
    Loads dataset, filters specific attacks, scales numerical features,
    encodes categorical features, and balances the training set using SMOTE.

    Returns:
        Tuple containing split data (Train, Val, Test), metadata (cols), and encoders.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath).drop(columns=['label', 'id'])

    # Filter for specific attack categories to focus the study
    target_attacks = ['Normal', 'Exploits', 'Fuzzers', 'Generic']
    df = df[df['attack_cat'].isin(target_attacks)].copy()

    # Encode target labels
    le_target = LabelEncoder()
    df['attack_cat'] = le_target.fit_transform(df['attack_cat'])
    class_names = le_target.classes_

    X = df.drop(columns=['attack_cat'])
    y = df['attack_cat']

    # Stratified Split: Train (70%) / Val (15%) / Test (15%)
    # Stratification ensures all classes are represented in splits, crucial for imbalanced data.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Identify Feature Types
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # 1. Scale Numerical Features (Standardization)
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 2. Encode Categorical Features
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on ALL data to prevent "Unseen Label" errors during validation/testing
        # In production, we would handle unknown tokens explicitly (e.g., using <UNK>).
        all_values = pd.concat([X_train[col], X_val[col], X_test[col]]).astype(str)
        le.fit(all_values)
        encoders[col] = le

        X_train[col] = le.transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    # 3. Apply SMOTE (Synthetic Minority Over-sampling Technique)
    # This generates synthetic samples for minority classes to prevent bias towards the majority class.
    print("Applying SMOTE to balance training classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Reconstruct DataFrame to preserve column names for the model input
    X_train = pd.DataFrame(X_train_res, columns=X.columns)
    y_train = pd.Series(y_train_res)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_cols, cat_cols, encoders, class_names


def build_ft_transformer(
        input_cols: List[str],
        cat_cols: List[str],
        encoders: Dict,
        config: Dict
) -> Model:
    """
    Constructs the FT-Transformer architecture.
    FIXED: Replaced non-existent layers.Stack with Reshape + Concatenate.
    """
    inputs = {}
    embeddings = []

    # --- Feature Tokenizer Block ---
    for col in input_cols:
        inputs[col] = Input(shape=(1,), name=col)
        if col in cat_cols:
            # Embedding for categorical features
            vocab_size = len(encoders[col].classes_) + 1
            emb = layers.Embedding(vocab_size, config["EMBEDDING_DIM"])(inputs[col])
            # FIX: Reshape to (1, dim) to prepare for concatenation
            embeddings.append(layers.Reshape((1, config["EMBEDDING_DIM"]))(emb))
        else:
            # Linear projection for numerical features
            emb = layers.Dense(config["EMBEDDING_DIM"])(inputs[col])
            emb = layers.LayerNormalization(epsilon=1e-6)(emb)
            # FIX: Reshape to (1, dim) to prepare for concatenation
            embeddings.append(layers.Reshape((1, config["EMBEDDING_DIM"]))(emb))

    # FIX: Use Concatenate to stack along the time axis (axis 1)
    # Output shape: (batch_size, num_features, embedding_dim)
    x = layers.Concatenate(axis=1)(embeddings)

    # Add the Learnable [CLS] Token
    x = AddCLSToken(config["EMBEDDING_DIM"])(x)

    # --- Transformer Encoder Blocks ---
    for _ in range(config["NUM_BLOCKS"]):
        # Part 1: Multi-Head Self-Attention
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(
            num_heads=config["NUM_HEADS"],
            key_dim=config["EMBEDDING_DIM"] // config["NUM_HEADS"],
            dropout=config["DROPOUT"]
        )(x_norm, x_norm)
        x = layers.Add()([x, attn_output])

        # Part 2: Feed-Forward Network
        x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
        ffn = layers.Dense(config["FFN_DIM"], activation='relu', kernel_regularizer=regularizers.l2(config["L2_REG"]))(
            x_norm)
        ffn = layers.Dense(config["EMBEDDING_DIM"])(ffn)
        x = layers.Add()([x, layers.Dropout(config["DROPOUT"])(ffn)])

    # --- Classification Head ---
    cls_output = x[:, 0, :]

    x_head = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(config["L2_REG"]))(cls_output)
    x_head = layers.Dropout(config["DROPOUT"])(x_head)
    output = layers.Dense(config["NUM_CLASSES"], activation='softmax')(x_head)

    return Model(inputs=inputs, outputs=output)


def df_to_dict(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Helper to convert DataFrame columns into a dictionary of numpy arrays.
    This format is significantly faster for Keras Multi-Input models than passing a list.
    """
    return {col: df[col].values for col in df.columns}


def plot_history(histories: List[Dict], save_path: str):
    """Generates and saves training curves for Loss and Accuracy across multiple runs."""
    plt.figure(figsize=(12, 5))
    metrics = ['accuracy', 'loss']

    for idx, metric in enumerate(metrics):
        plt.subplot(1, 2, idx + 1)
        for i, h in enumerate(histories):
            plt.plot(h[metric], alpha=0.4, label=f'Run {i + 1} Train')
            plt.plot(h[f'val_{metric}'], alpha=0.8, linestyle='--', label=f'Run {i + 1} Val')

        plt.title(f'Model {metric.capitalize()}')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training metrics plot saved to: {save_path}")


def main():
    if not os.path.exists(CONFIG["PATH"]):
        raise FileNotFoundError(f"Dataset not found at {CONFIG['PATH']}. Please check the path.")

    # 1. Load and Preprocess Data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), num_cols, cat_cols, encoders, classes = load_and_preprocess(
        CONFIG["PATH"])

    # Prepare Inputs
    train_data = (df_to_dict(X_train), keras.utils.to_categorical(y_train, CONFIG["NUM_CLASSES"]))
    val_data = (df_to_dict(X_val), keras.utils.to_categorical(y_val, CONFIG["NUM_CLASSES"]))
    test_data = (df_to_dict(X_test), keras.utils.to_categorical(y_test, CONFIG["NUM_CLASSES"]))

    # Storage for aggregation
    all_histories = []
    all_results = []  # Store [loss, accuracy, precision, recall, auc]
    all_reports = []  # Store dicts from classification_report

    # 2. Training Loop (3 Runs)
    print("\nStarting Training Pipeline (3 Independent Runs)...")

    for run in range(3):
        print(f"\n=== Run {run + 1}/3 ===")
        keras.utils.set_random_seed(run)

        model = build_ft_transformer(X_train.columns, cat_cols, encoders, CONFIG)

        # UPDATED: Added Precision, Recall, and AUC to metrics so we can aggregate them later
        optimizer = keras.optimizers.AdamW(learning_rate=CONFIG["LR"], weight_decay=1e-4)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(CONFIG['DIRS']['models'], f"ft_transformer_run_{run}.keras"),
                monitor='val_loss',
                save_best_only=True
            )
        ]

        start_time = time.time()
        history = model.fit(
            train_data[0], train_data[1],
            validation_data=val_data,
            epochs=CONFIG["EPOCHS"],
            batch_size=CONFIG["BATCH_SIZE"],
            callbacks=callbacks,
            verbose=1
        )
        print(f"Run time: {time.time() - start_time:.2f}s")
        all_histories.append(history.history)

        # 3. Evaluation per Run
        # Get scalar metrics (Loss, Acc, Prec, Rec, AUC)
        scores = model.evaluate(test_data[0], test_data[1], verbose=0)
        all_results.append(scores)

        # Get detailed classification report
        y_pred_prob = model.predict(test_data[0], verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Save as dictionary for aggregation
        report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        all_reports.append(report_dict)

        # Print simple report for logs
        print(classification_report(y_test, y_pred, target_names=classes))

    # --- 4. Aggregation Block (Added based on your request) ---
    results_np = np.array(all_results)

    print("\n=== Aggregated Results (3 Runs) ===")
    # metrics_names matches the order in model.compile: [loss, accuracy, precision, recall, auc]
    print(f"Mean Metrics (Loss, Acc, Prec, Rec, AUC): {np.mean(results_np, axis=0)}")
    print(f"Std Deviation: {np.std(results_np, axis=0)}")

    # Aggregate classification_report
    metrics_per_class = {name: {'precision': [], 'recall': [], 'f1-score': []} for name in classes}

    for report in all_reports:
        for name in classes:
            metrics_per_class[name]['precision'].append(report[name]['precision'])
            metrics_per_class[name]['recall'].append(report[name]['recall'])
            metrics_per_class[name]['f1-score'].append(report[name]['f1-score'])

    print("\n=== Detailed Aggregated Report ===")
    for name in classes:
        print(f"\nClass: {name}")
        print(
            f"  Precision: {np.mean(metrics_per_class[name]['precision']):.4f} ± {np.std(metrics_per_class[name]['precision']):.4f}")
        print(
            f"  Recall:    {np.mean(metrics_per_class[name]['recall']):.4f} ± {np.std(metrics_per_class[name]['recall']):.4f}")
        print(
            f"  F1-score:  {np.mean(metrics_per_class[name]['f1-score']):.4f} ± {np.std(metrics_per_class[name]['f1-score']):.4f}")

    # 5. Visualization
    plot_history(all_histories, os.path.join(CONFIG['DIRS']['images'], 'training_metrics.png'))


if __name__ == "__main__":
    main()