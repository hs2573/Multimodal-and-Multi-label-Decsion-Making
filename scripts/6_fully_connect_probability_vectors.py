"""
fully_connect_probability_vectors.py

Concatenate probability vectors from sleep stage (R, S1, S2, S3, S4, W) and disorder (B, I, N, Na, Nf, P, Rb, S)
classification to create second-stage combined train and test CSV files.
Assumes data is aligned (same sample count and order) from connecting_probability_vectors.py.
Uses pd.concat without for loops for data loading.

Dependencies: pandas, numpy, os
"""

import os
import pandas as pd
import numpy as np

# Configuration
PROB_DIR = ".\results\probability vector"

def fully_connect_probability_vectors():
    """Concatenate stage and disorder probability vectors for train and test subsets."""
    # --- Train Data ---
    # Load train data for stage and disorder
    train_data_stage = pd.read_csv(".\results\probability vector\\train_data_stage.csv")
    train_data_disorder = pd.read_csv(".\results\probability vector\\train_data_disorder.csv")
    
    # Concatenate stage and disorder
    train_data_combined = pd.concat([train_data_stage, train_data_disorder], axis=1)
    
    # Load and save labels
    train_label_stage = pd.read_csv(".\results\probability vector\\train_label_stage.csv")
    train_label_disorder = pd.read_csv(".\results\probability vector\\train_label_disorder.csv")
    
    # Save
    train_data_combined.to_csv(".\results\probability vector\\train_data_combined.csv", index=False)
    train_label_stage.to_csv(".\results\probability vector\\train_label_combined_stage.csv", index=False)
    train_label_disorder.to_csv(".\results\probability vector\\train_label_combined_disorder.csv", index=False)
    print("Saved combined train data to .\results\probability vector\\train_data_combined.csv")
    print("Saved combined train stage label to .\results\probability vector\\train_label_combined_stage.csv")
    print("Saved combined train disorder label to .\results\probability vector\\train_label_combined_disorder.csv")
    
    # --- Test Data ---
    # Load test data for stage and disorder
    test_data_stage = pd.read_csv(".\results\probability vector\\test_data_stage.csv")
    test_data_disorder = pd.read_csv(".\results\probability vector\\test_data_disorder.csv")
    
    # Concatenate stage and disorder
    test_data_combined = pd.concat([test_data_stage, test_data_disorder], axis=1)
    
    # Load and save labels
    test_label_stage = pd.read_csv(".\results\probability vector\\test_label_stage.csv")
    test_label_disorder = pd.read_csv(".\results\probability vector\\test_label_disorder.csv")
    
    # Save
    test_data_combined.to_csv(".\results\probability vector\\test_data_combined.csv", index=False)
    test_label_stage.to_csv(".\results\probability vector\\test_label_combined_stage.csv", index=False)
    test_label_disorder.to_csv(".\results\probability vector\\test_label_combined_disorder.csv", index=False)
    print("Saved combined test data to .\results\probability vector\\test_data_combined.csv")
    print("Saved combined test stage label to .\results\probability vector\\test_label_combined_stage.csv")
    print("Saved combined test disorder label to .\results\probability vector\\test_label_combined_disorder.csv")

def main():
    """Main function to concatenate stage and disorder probability vectors."""
    # Ensure output directory exists
    os.makedirs(PROB_DIR, exist_ok=True)
    
    fully_connect_probability_vectors()

if __name__ == "__main__":
    main()