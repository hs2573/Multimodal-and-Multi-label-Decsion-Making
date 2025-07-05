"""
batch_generate_spectrograms.py

Generate linear and logarithmic spectrograms for multiple EDF files using STFT.
Processes 30-second and 10-second EEG/EMG/ECG segments from PhysioNet CAP Sleep Database,
as described in *Sensors 2023, 23, 3468* (Section 2.1-2.2).

Dependencies: numpy, pandas, matplotlib, mne, fileinput, scipy
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
TXT_DIR = os.path.join(BASE_DIR, "txt_data")
SAVE_DIR_30S = os.path.join(BASE_DIR, "img_channel_30s")
SAVE_DIR_10S = os.path.join(BASE_DIR, "img_channel_10s")
FILE_NUMBERS = range(11, 17)  # Process n11 to n16
NFFT = 1024
FS = 50  # Sampling frequency (Hz) for EKG channels
NOVERLAP = 990  # Fixed overlap for STFT

def clean_txt_file(txt_file, txt_dir):
    """
    Clean txt file by replacing specific strings for consistent parsing.
    
    Args:
        txt_file (str): Name of txt file (e.g., 'n11.txt').
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

def normalize_channel_name(channel):
    """
    Normalize channel names to consistent format.
    
    Args:
        channel (str): Raw channel name (e.g., 'LOC / A2').
    
    Returns:
        str: Normalized channel name (e.g., 'LOC-A2').
    """
    channel_map = {
        "LOC / A2": "LOC-A2",
        "ROC / A1": "ROC-A1",
        "C4A1": "C4-A1",
        "F3A2": "F3-A2",
        "F4A1": "F4-A1",
        "EKG": "ekg",
        "O1A2": "O1-A2",
        "O2A1": "O2-A1",
        "C3A2": "C3-A2",
        "CHIN-1": "CHIN1",
        "CHIN-0": "CHIN0",
        "DX1-DX2": "Dx1-DX2",
        "Tib dx": "tib dx",
        "flow": "Flow"
    }
    return channel_map.get(channel, channel)

def load_labels(txt_file, txt_dir):
    """
    Load and process labels from txt file, extracting 30-second and 10-second segments.
    
    Args:
        txt_file (str): Name of txt file (e.g., 'n11.txt').
        txt_dir (str): Directory containing txt files.
    
    Returns:
        pd.DataFrame: DataFrame with start/end times for 30s and 10s segments and labels.
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
    
    # Generate 10-second segments
    df['st10_1'] = df['st']
    df['ed10_1'] = df['st'] + 10
    df['st10_2'] = df['st'] + 10
    df['ed10_2'] = df['st'] + 20
    df['st10_3'] = df['st'] + 20
    df['ed10_3'] = df['st'] + 30
    
    return df[['st', 'ed', 'st10_1', 'ed10_1', 'st10_2', 'ed10_2', 'st10_3', 'ed10_3', 'ST']]

def main():
    """Batch process multiple EDF files to generate 30s and 10s spectrograms."""
    db_df = pd.read_csv(os.path.join(BASE_DIR, "db_n.csv"), index_col=0)
    
    for f_number in FILE_NUMBERS:
        txt_file = f"n{f_number}.txt"
        edf_file = f"n{f_number}.edf"
        
        # Clean txt file
        clean_txt_file(txt_file, TXT_DIR)
        
        # Load labels
        df = load_labels(txt_file, TXT_DIR)
        
        # Load EDF data
        data = mne.io.read_raw_edf(os.path.join(DATA_DIR, edf_file), preload=True)
        info = data.info
        channels = [normalize_channel_name(ch) for ch in data.ch_names]
        picks = mne.pick_types(info, eeg=True, eeg=False, eog=True, ecg=True, exclude='bads')
        
        # Process each segment
        for i in range(len(df)):
            st, et, label = df['st'].iloc[i], df['ed'].iloc[i], df['ST'].iloc[i]
            t_idx = data.time_as_index([st, et])
            data_p, _ = data[picks, t_idx[0]:t_idx[1]]
            
            # 30-second spectrogram
            for j, channel in enumerate(channels):
                vmin, vmax = db_df.get(channel, [None, None])
                generate_spectrogram(
                    data=data_p[j, :],
                    channel=channel,
                    save_location=SAVE_DIR_30S,
                    file=f"n{f_number}",
                    st=st,
                    et=et,
                    label=label,
                    Fs=FS,
                    NFFT=NFFT,
                    noverlap=NOVERLAP/1024,
                    vmin=vmin,
                    vmin=vmax
                )
            
            # 10-second spectrograms
            for seg in [(df['st10_1'].iloc[i], df['ed10_1'].iloc[i]),
                        (df['st10_2'].iloc[i], df['ed10_2'].iloc[i]),
                        (df['st10_3'].iloc[i], df['ed10_3'].iloc[i])]:
                st10, et10 = seg
                t_idx = data.time_as_index([st10, et10])
                data_p, _ = data[picks, t_idx[0]:t_idx[1]]
                
                for j, channel in enumerate(channels):
                    vmin, vmax = db_df.get(channel, [None, None])
                    generate_spectrogram(
                        data=data_p[j, :],
                        channel=channel,
                        save_location=SAVE_DIR_10S,
                        file=f"n{f_number}",
                        st=st10,
                        et=et10,
                        label=label,
                        Fs=FS,
                        NFFT=NFFT,
                        noverlap=NOVERLAP/1024,
                        vmin=vmin,
                        vmax=vmax
                    )
        
        del data, df
        import gc
        gc.collect()

if __name__ == "__main__":
    main()