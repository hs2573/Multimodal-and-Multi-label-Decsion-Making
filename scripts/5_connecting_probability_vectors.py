"""
connecting_probability_vectors.py

Concatenate probability vectors from EEG, ECG, EMG modalities for sleep stage (R, S1, S2, S3, S4, W)
and disorder (B, I, N, Na, Nf, P, Rb, S) classification to create second-stage train and test CSV files.
Assumes data is aligned (same sample count and order) from train_{disorder,stage}_single_modality.py.
Uses pd.concat without for loops for data loading.

Dependencies: pandas, numpy, os
"""

import os
import pandas as pd
import numpy as np

# Configuration
PROB_DIR = ".\results\probability vector"
STAGE_CLASSES = ['R', 'S1', 'S2', 'S3', 'S4', 'W']
DISORDER_CLASSES = ['B', 'I', 'N', 'Na', 'Nf', 'P', 'Rb', 'S']

def connect_probability_vectors():
    """Concatenate probability vectors for EEG, ECG, EMG modalities for train and test subsets."""
    # --- Sleep Stage (train) ---
    # Load train data
    eeg_train_data_stage = pd.read_csv(".\results\probability vector\EEG_train_data_stage.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    ecg_train_data_stage = pd.read_csv(".\results\probability vector\ECG_train_data_stage.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    emg_train_data_stage = pd.read_csv(".\results\probability vector\EMG_train_data_stage.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    
    # Rename columns
    eeg_train_data_stage.columns = [f"{cls}_E" for cls in STAGE_CLASSES]
    ecg_train_data_stage.columns = [f"{cls}_C" for cls in STAGE_CLASSES]
    emg_train_data_stage.columns = [f"{cls}_M" for cls in STAGE_CLASSES]
    
    # Concatenate
    train_data_stage = pd.concat([eeg_train_data_stage, ecg_train_data_stage, emg_train_data_stage], axis=1)
    
    # Load and save label (using EEG as labels are identical)
    train_label_stage = pd.read_csv(".\results\probability vector\EEG_train_label_stage.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    train_label_stage.columns = ['label']
    
    # Save
    train_data_stage.to_csv(".\results\probability vector\\train_data_stage.csv", index=False)
    train_label_stage.to_csv(".\results\probability vector\\train_label_stage.csv", index=False)
    print("Saved merged stage train data to .\results\probability vector\\train_data_stage.csv")
    print("Saved merged stage train label to .\results\probability vector\\train_label_stage.csv")
    
    # --- Sleep Stage (test) ---
    # Load test data
    eeg_test_data_stage = pd.read_csv(".\results\probability vector\EEG_test_data_stage.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    ecg_test_data_stage = pd.read_csv(".\results\probability vector\ECG_test_data_stage.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    emg_test_data_stage = pd.read_csv(".\results\probability vector\EMG_test_data_stage.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    
    # Rename columns
    eeg_test_data_stage.columns = [f"{cls}_E" for cls in STAGE_CLASSES]
    ecg_test_data_stage.columns = [f"{cls}_C" for cls in STAGE_CLASSES]
    emg_test_data_stage.columns = [f"{cls}_M" for cls in STAGE_CLASSES]
    
    # Concatenate
    test_data_stage = pd.concat([eeg_test_data_stage, ecg_test_data_stage, emg_test_data_stage], axis=1)
    
    # Load and save label
    test_label_stage = pd.read_csv(".\results\probability vector\EEG_test_label_stage.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    test_label_stage.columns = ['label']
    
    # Save
    test_data_stage.to_csv(".\results\probability vector\\test_data_stage.csv", index=False)
    test_label_stage.to_csv(".\results\probability vector\\test_label_stage.csv", index=False)
    print("Saved merged stage test data to .\results\probability vector\\test_data_stage.csv")
    print("Saved merged stage test label to .\results\probability vector\\test_label_stage.csv")
    
    # --- Sleep Disorder (train) ---
    # Load train data
    eeg_train_data_disorder = pd.read_csv(".\results\probability vector\EEG_train_data_disorder.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    ecg_train_data_disorder = pd.read_csv(".\results\probability vector\ECG_train_data_disorder.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    emg_train_data_disorder = pd.read_csv(".\results\probability vector\EMG_train_data_disorder.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    
    # Rename columns
    eeg_train_data_disorder.columns = [f"{cls}_E" for cls in DISORDER_CLASSES]
    ecg_train_data_disorder.columns = [f"{cls}_C" for cls in DISORDER_CLASSES]
    emg_train_data_disorder.columns = [f"{cls}_M" for cls in DISORDER_CLASSES]
    
    # Concatenate
    train_data_disorder = pd.concat([eeg_train_data_disorder, ecg_train_data_disorder, emg_train_data_disorder], axis=1)
    
    # Load and save label
    train_label_disorder = pd.read_csv(".\results\probability vector\EEG_train_label_disorder.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    train_label_disorder.columns = ['label']
    
    # Save
    train_data_disorder.to_csv(".\results\probability vector\\train_data_disorder.csv", index=False)
    train_label_disorder.to_csv(".\results\probability vector\\train_label_disorder.csv", index=False)
    print("Saved merged disorder train data to .\results\probability vector\\train_data_disorder.csv")
    print("Saved merged disorder train label to .\results\probability vector\\train_label_disorder.csv")
    
    # --- Sleep Disorder (test) ---
    # Load test data
    eeg_test_data_disorder = pd.read_csv(".\results\probability vector\EEG_test_data_disorder.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    ecg_test_data_disorder = pd.read_csv(".\results\probability vector\ECG_test_data_disorder.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    emg_test_data_disorder = pd.read_csv(".\results\probability vector\EMG_test_data_disorder.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    
    # Rename columns
    eeg_test_data_disorder.columns = [f"{cls}_E" for cls in DISORDER_CLASSES]
    ecg_test_data_disorder.columns = [f"{cls}_C" for cls in DISORDER_CLASSES]
    emg_test_data_disorder.columns = [f"{cls}_M" for cls in DISORDER_CLASSES]
    
    # Concatenate
    test_data_disorder = pd.concat([eeg_test_data_disorder, ecg_test_data_disorder, emg_test_data_disorder], axis=1)
    
    # Load and save label
    test_label_disorder = pd.read_csv(".\results\probability vector\EEG_test_label_disorder.csv").drop("Unnamed: 0", axis=1, errors='ignore')
    test_label_disorder.columns = ['label']
    
    # Save
    test_data_disorder.to_csv(".\results\probability vector\\test_data_disorder.csv", index=False)
    test_label_disorder.to_csv(".\results\probability vector\\test_label_disorder.csv", index=False)
    print("Saved merged disorder test data to .\results\probability vector\\test_data_disorder.csv")
    print("Saved merged disorder test label to .\results\probability vector\\test_label_disorder.csv")

def main():
    """Main function to concatenate probability vectors."""
    # Ensure output directory exists
    os.makedirs(PROB_DIR, exist_ok=True)
    
    connect_probability_vectors()

if __name__ == "__main__":
    main()