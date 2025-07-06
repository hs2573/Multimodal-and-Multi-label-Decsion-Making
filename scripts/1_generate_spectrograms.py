"""
generate_spectrograms.py

Generate linear and logarithmic spectrograms from a single EDF file using STFT.
Processes 30-second EEG/EMG/ECG segments from PhysioNet CAP Sleep Database,
as described in *Sensors 2023, 23, 3468* (Section 2.2).

Dependencies: numpy, pandas, matplotlib, mne, fileinput
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import fileinput
from utils import generate_spectrogram

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TXT_DIR = os.path.join(BASE_DIR, "text_data")
SAVE_DIR = os.path.join(BASE_DIR, "img_channel")
FILE_NAME = "n1"  # Example EDF/txt file (without extension)
NFFT = 1024
FS = 512  # Sampling frequency (Hz)
NOVERLAP = int(NFFT * 0.9)  # 90% overlap

def clean_txt_file(txt_file, txt_dir):
    """
    Clean txt file by replacing specific strings for consistent parsing.
    
    Args:
        txt_file (str): Name of txt file (e.g., 'n1.txt').
        txt_dir (str): Directory containing txt files.
    """
    filename = os.path.join(txt_dir, txt_file)
    replacements = [
        ("Sleep Stage", "Sleep_Stage"),
        ("Unknown Position", "Unknown_Position"),
        ("Time [hh:mm:ss]", "Time_[hh:mm:ss]")
    ]
    for old, new in replacements:
        with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
            for line in file:
                print(line.replace(old, new), end='')

def load_labels(txt_file, txt_dir):
    """
    Load and process labels from txt file, extracting 30-second segments.
    
    Args:
        txt_file (str): Name of txt file (e.g., 'n1.txt').
        txt_dir (str): Directory containing txt files.
    
    Returns:
        pd.DataFrame: DataFrame with start time (st), end time (ed), and labels (ST).
    """
    file_id = txt_file[-6:-4]
    names = ('ST', 'TIM', 'EVE', 'DUR', 'LO') if file_id in ['n2', 'n3', 'n6', 'n8'] else ('ST', 'PO', 'TIM', 'EVE', 'DUR', 'LO')
    df = pd.read_csv(os.path.join(txt_dir, txt_file), header=None, delim_whitespace=True, names=names).iloc[19:, :]
    df = df.reset_index(drop=True)
    df['DUR'] = pd.to_numeric(df['DUR'])
    df = df[df['DUR'] == 30].reset_index(drop=True)
    
    df['TIM'] = pd.to_datetime(df['TIM'], format='%H:%M:%S')
    df['SEC'] = (df['TIM'].dt.hour * 3600 + df['TIM'].dt.minute * 60 + df['TIM'].dt.second)
    first_sec = df['SEC'].iloc[0]
    df['st'] = df['SEC'].apply(lambda x: x - first_sec if x >= first_sec else 86400 + x - first_sec)
    df['ed'] = df['st'] + 30
    
    return df[['st', 'ed', 'ST']]

def main():
    """Process a single EDF file to generate spectrograms for multiple channels."""
    # Clean txt file
    txt_file = f"{FILE_NAME}.txt"
    clean_txt_file(txt_file, TXT_DIR)
    
    # Load labels
    df = load_labels(txt_file, TXT_DIR)
    
    # Load EDF data
    data = mne.io.read_raw_edf(os.path.join(DATA_DIR, f"{FILE_NAME}.edf"), preload=True)
    info = data.info
    channels = data.ch_names
    picks = mne.pick_types(info, eeg=True, exclude='bads')
    
    # Process each 30-second segment
    for i in range(len(df)):
        st, et, label = df['st'].iloc[i], df['ed'].iloc[i], df['ST'].iloc[i]
        t_idx = data.time_as_index([st, et])
        data_p, _ = data[picks, t_idx[0]:t_idx[1]]
        
        for j, channel in enumerate(channels):
            generate_spectrogram(
                data=data_p[j, :],
                channel=channel,
                save_location=SAVE_DIR,
                file=FILE_NAME,
                st=st,
                et=et,
                label=label,
                Fs=FS,
                NFFT=NFFT,
                noverlap=NOVERLAP
            )

if __name__ == "__main__":
    main()
