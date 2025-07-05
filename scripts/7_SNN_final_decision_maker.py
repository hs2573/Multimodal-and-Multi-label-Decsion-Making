"""
SNN_final_decision_maker.py

Train shallow neural networks for sleep stage (R, S1, S2, S3, S4, W) and disorder (B, I, N, Na, Nf, P, Rb, S)
classification using combined probability vectors from fully_connect_probability_vectors.py.
Uses 42-dimensional input (18 stage + 24 disorder), outputs separate models for stage and disorder.

Dependencies: pandas, numpy, sklearn, keras
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical

# Configuration
PROB_DIR = ".\results\probability vector"
STAGE_CLASSES = ['R', 'S1', 'S2', 'S3', 'S4', 'W']
DISORDER_CLASSES = ['B', 'I', 'N', 'Na', 'Nf', 'P', 'Rb', 'S']

def build_model(input_dim, output_dim):
    """Build a shallow neural network model."""
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def train_final_decision_maker():
    """Train shallow NN for stage and disorder classification."""
    # --- Load and preprocess data ---
    # Train data
    train_data_combined = pd.read_csv(".\results\probability vector\\train_data_combined.csv")
    train_label_stage = pd.read_csv(".\results\probability vector\\train_label_combined_stage.csv")
    train_label_disorder = pd.read_csv(".\results\probability vector\\train_label_combined_disorder.csv")
    
    # Test data
    test_data_combined = pd.read_csv(".\results\probability vector\\test_data_combined.csv")
    test_label_stage = pd.read_csv(".\results\probability vector\\test_label_combined_stage.csv")
    test_label_disorder = pd.read_csv(".\results\probability vector\\test_label_combined_disorder.csv")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data_combined)
    X_test = scaler.transform(test_data_combined)
    
    # Convert labels to one-hot encoding
    y_train_stage = to_categorical(train_label_stage['label'], num_classes=len(STAGE_CLASSES))
    y_test_stage = to_categorical(test_label_stage['label'], num_classes=len(STAGE_CLASSES))
    y_train_disorder = to_categorical(train_label_disorder['label'], num_classes=len(DISORDER_CLASSES))
    y_test_disorder = to_categorical(test_label_disorder['label'], num_classes=len(DISORDER_CLASSES))
    
    # --- Stage Model ---
    stage_model = build_model(input_dim=42, output_dim=len(STAGE_CLASSES))
    
    # Callbacks
    checkpoint_stage = ModelCheckpoint(".\results\second_level_model\\best_model_stage.h5",
                                      monitor='val_accuracy', verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)
    early_stage = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
    
    # Train
    stage_model.fit(X_train, y_train_stage, epochs=100, batch_size=7,
                    validation_data=(X_test, y_test_stage), callbacks=[checkpoint_stage, early_stage], verbose=1)
    
    # Evaluate
    scores_stage_train = stage_model.evaluate(X_train, y_train_stage, batch_size=7, verbose=0)
    print(f"Stage Train Accuracy: {scores_stage_train[1]*100:.2f}%")
    scores_stage_test = stage_model.evaluate(X_test, y_test_stage, batch_size=7, verbose=0)
    print(f"Stage Test Accuracy: {scores_stage_test[1]*100:.2f}%")
    
    # Predict and report
    y_pred_stage = np.argmax(stage_model.predict(X_test), axis=1)
    y_test_true_stage = test_label_stage['label']
    print("Stage Confusion Matrix (Test):")
    print(confusion_matrix(y_test_true_stage, y_pred_stage))
    print("Stage Classification Report (Test):")
    print(classification_report(y_test_true_stage, y_pred_stage, target_names=STAGE_CLASSES))
    
    # --- Disorder Model ---
    disorder_model = build_model(input_dim=42, output_dim=len(DISORDER_CLASSES))
    
    # Callbacks
    checkpoint_disorder = ModelCheckpoint(".\results\second_level_model\\best_model_disorder.h5",
                                         monitor='val_accuracy', verbose=1, save_best_only=True,
                                         save_weights_only=False, mode='auto', period=1)
    early_disorder = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
    
    # Train
    disorder_model.fit(X_train, y_train_disorder, epochs=100, batch_size=7,
                      validation_data=(X_test, y_test_disorder), callbacks=[checkpoint_disorder, early_disorder], verbose=1)
    
    # Evaluate
    scores_disorder_train = disorder_model.evaluate(X_train, y_train_disorder, batch_size=7, verbose=0)
    print(f"Disorder Train Accuracy: {scores_disorder_train[1]*100:.2f}%")
    scores_disorder_test = disorder_model.evaluate(X_test, y_test_disorder, batch_size=7, verbose=0)
    print(f"Disorder Test Accuracy: {scores_disorder_test[1]*100:.2f}%")
    
    # Predict and report
    y_pred_disorder = np.argmax(disorder_model.predict(X_test), axis=1)
    y_test_true_disorder = test_label_disorder['label']
    print("Disorder Confusion Matrix (Test):")
    print(confusion_matrix(y_test_true_disorder, y_pred_disorder))
    print("Disorder Classification Report (Test):")
    print(classification_report(y_test_true_disorder, y_pred_disorder, target_names=DISORDER_CLASSES))

def main():
    """Main function to train final decision-making models."""
    # Ensure output directory exists
    os.makedirs(PROB_DIR, exist_ok=True)
    
    train_final_decision_maker()

if __name__ == "__main__":
    main()