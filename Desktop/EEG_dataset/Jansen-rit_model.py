import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Sigmoid activation function
def S(v, e0=2.5, v0=6.0, r=0.56):
    return 2 * e0 / (1 + np.exp(r * (v0 - v)))

# Model parameters
params = {
    "A": 3.25,
    "a": 100.0,
    "B": 22.0,
    "b": 50.0,
    "C": 135.0,
    "C1": 135.0,
    "C2": 0.8 * 135.0,
    "C3": 0.25 * 135.0,
    "C4": 0.25 * 135.0
}

# Jansen-Rit model ODEs
def jansen_rit(t, y, p_input, pars=params):
    y0, y3, y1, y4, y2, y5 = y
    A, a = pars["A"], pars["a"]
    B, b = pars["B"], pars["b"]
    C1, C2, C3, C4 = pars["C1"], pars["C2"], pars["C3"], pars["C4"]

    dydt = np.zeros_like(y)
    dydt[0] = y3
    dydt[1] = A * a * S(y1 - y2) - 2 * a * y3 - a**2 * y0
    dydt[2] = y4
    dydt[3] = A * a * (p_input + C2 * S(C1 * y0)) - 2 * a * y4 - a**2 * y1
    dydt[4] = y5
    dydt[5] = B * b * C4 * S(C3 * y0) - 2 * b * y5 - b**2 * y2

    return dydt

# Simulation settings
t_start, t_end = 0, 1      # seconds
fs = 1000                  # Hz
t_eval = np.linspace(t_start, t_end, int(fs * (t_end - t_start)))

# External input: random noise around 120 Hz
np.random.seed(0)
p_array = 120 + 30 * np.random.randn(len(t_eval))
p_func = interp1d(t_eval, p_array, kind='linear', fill_value="extrapolate")

# Wrapper for solve_ivp
def rhs(t, y):
    return jansen_rit(t, y, p_func(t))

# Initial state
y0 = np.zeros(6)

# Integrate ODEs
sol = solve_ivp(rhs, [t_start, t_end], y0, t_eval=t_eval, method='RK45')

# EEG-like signal from pyramidal cell outputs
y1, y2 = sol.y[2], sol.y[4]
eeg_signal = y1 - y2

# Plot the output
plt.figure(figsize=(10, 4))
plt.plot(sol.t, eeg_signal, label="EEG Output (y1 - y2)")
plt.title("Jansen–Rit Neural Mass Model Simulation")
plt.xlabel("Time (s)")
plt.ylabel("EEG Voltage (a.u.)")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("jansen_rit_eeg_output.png")
