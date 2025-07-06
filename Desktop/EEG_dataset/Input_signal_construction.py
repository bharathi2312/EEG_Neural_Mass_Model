from scipy.signal import hilbert
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load all 36 EEG dataset files (s00 to s35) into a list
eeg_data = []
for i in range(36):
    filename = fr"C:\Users\subscriber\Desktop\EEG_dataset\s{str(i).zfill(2)}.csv"
    eeg_data.append(pd.read_csv(filename))



'''# Plot all EEG datasets in a single window with subplots
fig, axes = plt.subplots(9, 4, figsize=(20, 18))  # 9 rows x 4 columns = 36
axes = axes.flatten()
for idx, eeg in enumerate(eeg_data):
    axes[idx].plot(eeg)
    axes[idx].set_title(f's{str(idx).zfill(2)}')
    axes[idx].set_xlabel('Sample')
    axes[idx].set_ylabel('Amplitude')
plt.tight_layout()
plt.show()'''

# Phase and Amplitude Extraction using Hilbert Transform


# Plot all signal_input outputs in the same window
plt.figure(figsize=(12, 8))
for idx, eeg in enumerate(eeg_data):
    analytic_signal = hilbert(eeg)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)
    signal_input = amplitude_envelope * np.cos(instantaneous_phase)
    plt.plot(signal_input, label=f's{str(idx).zfill(2)}')
plt.title('Signal Input for All EEG Datasets')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend(fontsize='small', ncol=3)
plt.tight_layout()
plt.show()

    