from scipy.signal import hilbert
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurolib as nl  # Jansen-Rit neural mass model from neurolib

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

# Initialize the Jansen-Rit model

model = nl.jansenrit.JansenRitModel()
# Set model parameters
model.params['N'] = 12               # number of EEG electrodes = number of nodes
model.params['duration'] = 5000           # 5 seconds
model.params['dt'] = 1.0                  # 1 ms time step

# Phase and Amplitude Extraction using Hilbert Transform

for idx, eeg in enumerate(eeg_data):
    analytic_signal = hilbert(eeg)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.angle(analytic_signal)
    signal_input = amplitude_envelope * np.cos(instantaneous_phase)
    model.external_input = signal_input  
    model.run()
    output = model.output
    # Plot output for each node
    time = np.arange(model.params['duration']) * model.params['dt']
    for i in range(model.params['N']):
        plt.plot(time, output[i], label=f'Node {i+1}')
    plt.legend()
    plt.title("Simulated EEG Activity at Electrodes")
    plt.xlabel("Time (ms)")
    plt.ylabel("EEG Signal")
    plt.show()


