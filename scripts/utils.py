# utils.py
import os
import matplotlib.pyplot as plt

def generate_spectrogram(data, channel, save_location, file, st, et, label, Fs=512, NFFT=1024, noverlap=0.9, vmin=None, vmax=None):
    """
    Generate linear and logarithmic spectrograms using STFT.
    
    Args:
        data (np.ndarray): EEG/EMG/ECG data segment.
        channel (str): Channel name (e.g., 'C4-A1').
        save_location (str): Output directory for spectrograms.
        file (str): EDF filename (e.g., 'n1').
        st (float): Start time (seconds).
        et (float): End time (seconds).
        label (str): Sleep stage or disorder label (e.g., 'W').
        Fs (int): Sampling frequency (default: 512 Hz).
        NFFT (int): FFT points (default: 1024).
        noverlap (float): Overlap ratio (default: 0.8).
        vmin (float): Minimum value for spectrogram scaling.
        vmax (float): Maximum value for spectrogram scaling.
    """
    # Create directories
    linear_dir = os.path.join(save_location, channel, "linear", label)
    log_dir = os.path.join(save_location, channel, "log", label)
    os.makedirs(linear_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    plt.figure(figsize=(2.56, 2.56))
    plt.specgram(data, NFFT=NFFT, Fs=Fs, noverlap=int(NFFT * noverlap), cmap="jet", vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(os.path.join(linear_dir, f"{file}_{st}_{et}-{channel}.jpg"), 
                format="jpg", bbox_inches="tight")
    plt.yscale("symlog")
    plt.savefig(os.path.join(log_dir, f"{file}_{st}_{et}-{channel}.jpg"), 
                format="jpg", bbox_inches="tight")
    plt.clf()
    plt.close()
    
    
