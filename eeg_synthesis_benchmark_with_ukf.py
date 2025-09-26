"""
eeg_synthesis_benchmark_with_ukf.py

Pipeline:
- Generate synthetic multi-channel EEG samples using Jansen-Rit neural mass model.
- Save dataset samples as JSON in ./dataset/
- Run EKF baseline (kept for reference) and UKF baseline to estimate pyramidal states from noisy EEG
- Train a small GRU denoiser (PyTorch) to map noisy->clean EEG
- Evaluate and print RMSE and Pearson correlation metrics comparing baselines

Dependencies:
  pip install numpy scipy pandas torch tqdm

Author: Sindu Bharathi (extended: added UKF)
"""
import os
import json
import math
import random
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Jansen-Rit neural mass model (single cortical column)
# -------------------------------

def S(v: np.ndarray) -> np.ndarray:
    e0 = 2.5
    v0 = 6.0
    r = 0.56
    return 2*e0 / (1.0 + np.exp(r*(v0 - v)))

def jansen_rit_rhs(t: float, y: np.ndarray, p: Tuple[float, float, float, float, float, float, float, float, float]) -> np.ndarray:
    A,B,a,b,C,C1,C2,C3,C4 = p
    y0, y1, y2, y3, y4, y5 = y
    p_ext = 120.0 * (1 + 0.2 * np.sin(2*np.pi*10*t))
    S0 = S(y0 - y1 - y2)
    S1 = S(C2*y0)
    S2 = S(C4*y0)
    dy0 = y3
    dy1 = y4
    dy2 = y5
    dy3 = A * a * S0 - 2*a*y3 - (a*a)*y0
    dy4 = A * a * (p_ext + C1*S1) - 2*a*y4 - (a*a)*y1
    dy5 = B * b * C3 * S2 - 2*b*y5 - (b*b)*y2
    return np.array([dy0, dy1, dy2, dy3, dy4, dy5], dtype=float)

def simulate_column(params: Dict[str, float], t_span: Tuple[float,float], fs: float, y0=None) -> Tuple[np.ndarray, np.ndarray]:
    if y0 is None:
        y0 = np.zeros(6)
    t_eval = np.arange(t_span[0], t_span[1], 1.0/fs)
    p_tuple = (params['A'], params['B'], params['a'], params['b'],
               params['C'], params['C1'], params['C2'], params['C3'], params['C4'])
    sol = solve_ivp(jansen_rit_rhs, t_span, y0, t_eval=t_eval, args=(p_tuple,), rtol=1e-7, atol=1e-9)
    pyramidal = sol.y[0]
    return t_eval, pyramidal

# -------------------------------
# Dataset generation
# -------------------------------

def make_mixing_matrix(n_channels: int, n_columns: int, seed: int = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    M = rng.normal(loc=0.5, scale=0.3, size=(n_channels, n_columns))
    for i in range(min(n_channels, n_columns)):
        M[i, i] += 0.8
    return M

def generate_eeg_sample(sample_id: int, t_span=(0.0, 10.0), fs=250.0,
                        n_columns=3, n_channels=3, noise_std=0.5, seed=None) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed + sample_id)
        random.seed(seed + sample_id)
    A = 3.25; B = 22.0; a = 100.0; b = 50.0; C = 135.0
    C1 = C; C2 = 0.8*C; C3 = 0.25*C; C4 = 0.25*C
    columns = []
    column_params = []
    t_eval = None
    for i in range(n_columns):
        params = {'A': A * (1 + 0.05*(i - (n_columns-1)/2.0)), 'B': B, 'a': a, 'b': b, 'C': C, 'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4}
        column_params.append(params)
        y0 = np.random.normal(scale=0.1, size=6)
        t_eval, pyramidal = simulate_column(params, t_span, fs, y0=y0)
        columns.append(pyramidal)
    columns = np.vstack(columns)
    M = make_mixing_matrix(n_channels, n_columns, seed=(seed or 0)+1000)
    eeg_clean = M.dot(columns)
    noise = np.random.normal(scale=noise_std, size=eeg_clean.shape)
    eeg_noisy = eeg_clean + noise
    sample = {
        'id': f'eeg_sample_{sample_id:04d}',
        'n_channels': n_channels,
        'n_columns': n_columns,
        'fs': fs,
        'time': t_eval.tolist(),
        'eeg_noisy': eeg_noisy.tolist(),
        'eeg_clean': eeg_clean.tolist(),
        'columns_pyramidal': columns.tolist(),
        'mixing_matrix': M.tolist(),
        'params_per_column': column_params,
        'noise_std': noise_std
    }
    return sample

def build_dataset(n_samples: int = 100, out_dir: str = './dataset', seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    index_rows = []
    for i in trange(n_samples, desc='Generating dataset'):
        sample = generate_eeg_sample(i, seed=seed)
        fname = os.path.join(out_dir, sample['id'] + '.json')
        with open(fname, 'w') as f:
            json.dump(sample, f, indent=2)
        index_rows.append({'id': sample['id'], 'file': fname, 'n_channels': sample['n_channels'], 'fs': sample['fs']})
    df = pd.DataFrame(index_rows)
    df.to_csv(os.path.join(out_dir, 'index.csv'), index=False)
    print(f"Saved {n_samples} samples in {out_dir}")

# -------------------------------
# Numerical helpers
# -------------------------------

def rk4_step(fun, t, x, dt, *args):
    k1 = fun(t, x, *args)
    k2 = fun(t + dt/2.0, x + dt*k1/2.0, *args)
    k3 = fun(t + dt/2.0, x + dt*k2/2.0, *args)
    k4 = fun(t + dt, x + dt*k3, *args)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

def continuous_rhs_fullstate(t, X_flat, params_list):
    ncols = len(params_list)
    dX = np.zeros_like(X_flat)
    for i in range(ncols):
        y = X_flat[i*6:(i+1)*6]
        params = params_list[i]
        p_tuple = (params['A'], params['B'], params['a'], params['b'],
                   params['C'], params['C1'], params['C2'], params['C3'], params['C4'])
        d = jansen_rit_rhs(t, y, p_tuple)
        dX[i*6:(i+1)*6] = d
    return dX

# -------------------------------
# EKF baseline (kept for reference)
# -------------------------------

def ekf_assimilate(sample: Dict[str, Any], Q_scale: float = 1e-5, R_scale: float = 0.25) -> Dict[str, np.ndarray]:
    eeg_noisy = np.array(sample['eeg_noisy'])
    eeg_clean = np.array(sample['eeg_clean'])
    M = np.array(sample['mixing_matrix'])
    n_channels, n_time = eeg_noisy.shape
    n_columns = sample['n_columns']
    dt = 1.0 / sample['fs']
    params_list = sample['params_per_column']
    nx = 6 * n_columns
    x = np.zeros(nx)
    P = np.eye(nx) * 1e-2
    Q = np.eye(nx) * Q_scale
    R = np.eye(n_channels) * (R_scale)
    est_pyramidal = np.zeros((n_columns, n_time))
    H_full = np.zeros((n_channels, nx))
    for j in range(n_columns):
        H_full[:, j*6 + 0] = M[:, j]
    for ti in range(n_time):
        t = ti * dt
        x_pred = rk4_step(lambda tt, xx: continuous_rhs_fullstate(tt, xx, params_list), t, x, dt)
        eps = 1e-6
        F = np.zeros((nx, nx))
        f0 = continuous_rhs_fullstate(t, x, params_list)
        for k in range(nx):
            xp = x.copy(); xp[k] += eps
            fk = continuous_rhs_fullstate(t, xp, params_list)
            F[:, k] = (fk - f0) / eps
        Phi = np.eye(nx) + F*dt
        P_pred = Phi @ P @ Phi.T + Q
        yk = eeg_noisy[:, ti]
        S_mat = H_full @ P_pred @ H_full.T + R
        K = P_pred @ H_full.T @ np.linalg.inv(S_mat)
        innov = yk - H_full @ x_pred
        x = x_pred + K @ innov
        P = (np.eye(nx) - K @ H_full) @ P_pred
        for j in range(n_columns):
            est_pyramidal[j, ti] = x[j*6 + 0]
    eeg_rec = M.dot(est_pyramidal)
    def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
    rmse_channels = [rmse(eeg_rec[i], eeg_clean[i]) for i in range(n_channels)]
    results = {'est_pyramidal': est_pyramidal, 'eeg_rec': eeg_rec, 'rmse_channels': rmse_channels}
    return results

# -------------------------------
# UKF baseline (Merwe scaled sigma points)
# -------------------------------

def _merwe_weights(nx, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    Compute Merwe scaled weights for mean and covariance.
    """
    lam = alpha**2 * (nx + kappa) - nx
    c = nx + lam
    wm = np.full(2*nx + 1, 1.0/(2*c))
    wc = np.full(2*nx + 1, 1.0/(2*c))
    wm[0] = lam / c
    wc[0] = lam / c + (1 - alpha**2 + beta)
    return wm, wc, lam, c

def _sigma_points(x, P, c):
    """
    Compute sigma points for given state mean x and covariance P.
    Uses (2n+1) sigma points.
    """
    nx = x.size
    # numerical stability: add jitter if needed
    jitter = 1e-12
    try:
        sqrtP = np.linalg.cholesky(P + np.eye(nx)*jitter)
    except np.linalg.LinAlgError:
        # fallback: add more jitter
        sqrtP = np.linalg.cholesky(P + np.eye(nx)*1e-8)
    sqrt_cP = np.sqrt(c) * sqrtP
    X = np.zeros((2*nx + 1, nx))
    X[0] = x
    for i in range(nx):
        col = sqrt_cP[:, i]
        X[1 + i] = x + col
        X[1 + nx + i] = x - col
    return X  # shape (2n+1, n)

def ukf_assimilate(sample: Dict[str, Any], Q_scale: float = 1e-5, R_scale: float = 0.25,
                   alpha=1e-3, beta=2.0, kappa=0.0) -> Dict[str, np.ndarray]:
    """
    Unscented Kalman Filter assimilation for a single sample.
    Returns:
      - est_pyramidal: (n_columns, n_time)
      - eeg_rec: reconstructed EEG from estimated pyramidal
      - rmse_channels: per-channel RMSE vs clean EEG
    """
    eeg_noisy = np.array(sample['eeg_noisy'])
    eeg_clean = np.array(sample['eeg_clean'])
    M = np.array(sample['mixing_matrix'])
    n_channels, n_time = eeg_noisy.shape
    n_columns = sample['n_columns']
    dt = 1.0 / sample['fs']
    params_list = sample['params_per_column']
    nx = 6 * n_columns

    # UKF weights
    wm, wc, lam, c = _merwe_weights(nx, alpha=alpha, beta=beta, kappa=kappa)

    # initial mean & covariance
    x = np.zeros(nx)
    P = np.eye(nx) * 1e-2
    Q = np.eye(nx) * Q_scale
    R = np.eye(n_channels) * R_scale

    H_full = np.zeros((n_channels, nx))
    for j in range(n_columns):
        H_full[:, j*6 + 0] = M[:, j]

    est_pyramidal = np.zeros((n_columns, n_time))

    # time loop
    for ti in range(n_time):
        t = ti * dt
        # 1) Generate sigma points from current mean and P
        X_sigma = _sigma_points(x, P, c)  # shape (2n+1, nx)
        n_sigma = X_sigma.shape[0]
        # 2) Propagate each sigma point through process model (one discrete step via RK4)
        X_pred_sigma = np.zeros_like(X_sigma)
        for si in range(n_sigma):
            X_pred_sigma[si] = rk4_step(lambda tt, xx: continuous_rhs_fullstate(tt, xx, params_list), t, X_sigma[si], dt)
        # 3) Predict mean and covariance
        x_pred = np.zeros(nx)
        for si in range(n_sigma):
            x_pred += wm[si] * X_pred_sigma[si]
        P_pred = np.zeros((nx, nx))
        for si in range(n_sigma):
            dx = (X_pred_sigma[si] - x_pred).reshape(-1,1)
            P_pred += wc[si] * (dx @ dx.T)
        P_pred = P_pred + Q
        # 4) Predict observation sigma points and observation mean/cov
        Y_sigma = np.zeros((n_sigma, n_channels))
        for si in range(n_sigma):
            # observation: H_full @ X_pred_sigma[si]  (linear)
            Y_sigma[si] = H_full.dot(X_pred_sigma[si])
        y_pred = np.zeros(n_channels)
        for si in range(n_sigma):
            y_pred += wm[si] * Y_sigma[si]
        Pyy = np.zeros((n_channels, n_channels))
        Pxy = np.zeros((nx, n_channels))
        for si in range(n_sigma):
            dy = (Y_sigma[si] - y_pred).reshape(-1,1)
            dx = (X_pred_sigma[si] - x_pred).reshape(-1,1)
            Pyy += wc[si] * (dy @ dy.T)
            Pxy += wc[si] * (dx @ dy.T)
        Pyy = Pyy + R
        # 5) Kalman gain and update with measurement
        try:
            K = Pxy @ np.linalg.inv(Pyy)
        except np.linalg.LinAlgError:
            K = Pxy @ np.linalg.pinv(Pyy)
        yk = eeg_noisy[:, ti]
        innov = yk - y_pred
        x = x_pred + K.dot(innov)
        P = P_pred - K.dot(Pyy).dot(K.T)
        # ensure symmetry
        P = (P + P.T) / 2.0
        # store pyramidal
        for j in range(n_columns):
            est_pyramidal[j, ti] = x[j*6 + 0]
    eeg_rec = M.dot(est_pyramidal)
    def rmse(a,b): return np.sqrt(np.mean((a-b)**2))
    rmse_channels = [rmse(eeg_rec[i], eeg_clean[i]) for i in range(n_channels)]
    results = {'est_pyramidal': est_pyramidal, 'eeg_rec': eeg_rec, 'rmse_channels': rmse_channels}
    return results

# -------------------------------
# PyTorch GRU denoiser (unchanged)
# -------------------------------

class EEGDataset(Dataset):
    def __init__(self, index_csv: str, samples_dir: str, sample_ids: List[str]):
        self.samples = []
        for sid in sample_ids:
            fname = os.path.join(samples_dir, sid + '.json')
            with open(fname, 'r') as f:
                data = json.load(f)
            noisy = np.array(data['eeg_noisy']).astype(np.float32)
            clean = np.array(data['eeg_clean']).astype(np.float32)
            noisy_t = noisy.T
            clean_t = clean.T
            self.samples.append((noisy_t, clean_t))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        noisy, clean = self.samples[idx]
        return torch.from_numpy(noisy), torch.from_numpy(clean)

class GRUDenoiser(nn.Module):
    def __init__(self, n_channels: int, hidden_size: int = 128, n_layers: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.gru = nn.GRU(input_size=n_channels, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_channels)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out

def collate_pad(batch):
    noisy = torch.stack([b[0] for b in batch], dim=0)
    clean = torch.stack([b[1] for b in batch], dim=0)
    return noisy, clean

def train_gru(train_loader, val_loader, n_channels, epochs=10, lr=1e-3, device='cpu'):
    model = GRUDenoiser(n_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        count = 0
        for noisy, clean in train_loader:
            noisy = noisy.to(device); clean = clean.to(device)
            opt.zero_grad()
            pred = model(noisy)
            loss = loss_fn(pred, clean)
            loss.backward()
            opt.step()
            running += loss.item(); count += 1
        avg_train = running / max(1, count)
        model.eval()
        with torch.no_grad():
            running_v = 0.0; count_v = 0
            for noisy, clean in val_loader:
                noisy = noisy.to(device); clean = clean.to(device)
                pred = model(noisy)
                running_v += loss_fn(pred, clean).item(); count_v += 1
            avg_val = running_v / max(1, count_v)
        print(f"Epoch {ep}/{epochs} TrainLoss={avg_train:.6f} ValLoss={avg_val:.6f}")
    return model

# -------------------------------
# Utilities: evaluation metrics
# -------------------------------

def compute_metrics(eeg_est: np.ndarray, eeg_clean: np.ndarray) -> Dict[str, Any]:
    n_channels = eeg_clean.shape[0]
    rmse_list = []; corr_list = []
    for i in range(n_channels):
        a = eeg_est[i]; b = eeg_clean[i]
        rmse = np.sqrt(np.mean((a-b)**2))
        try:
            corr = pearsonr(a, b)[0]
        except Exception:
            corr = 0.0
        rmse_list.append(rmse); corr_list.append(corr)
    return {'rmse_per_channel': rmse_list, 'corr_per_channel': corr_list}

# -------------------------------
# Main driver: generate dataset, run EKF and UKF on a few samples, train GRU
# -------------------------------

def main():
    out_dir = './dataset'
    n_samples = 60
    print("Generating dataset...")
    build_dataset(n_samples=n_samples, out_dir=out_dir, seed=123)

    idx_df = pd.read_csv(os.path.join(out_dir, 'index.csv'))
    sample_ids = idx_df['id'].tolist()

    # EKF and UKF on first 5 samples
    print("\nRunning EKF and UKF baselines on first 5 samples...")
    ekf_metrics = []; ukf_metrics = []
    for sid in sample_ids[:5]:
        fname = os.path.join(out_dir, sid + '.json')
        with open(fname, 'r') as f:
            sample = json.load(f)
        ekf_res = ekf_assimilate(sample)
        ukf_res = ukf_assimilate(sample)
        ekf_m = compute_metrics(ekf_res['eeg_rec'], np.array(sample['eeg_clean']))
        ukf_m = compute_metrics(ukf_res['eeg_rec'], np.array(sample['eeg_clean']))
        print(f"{sid} EKF RMSE={ekf_m['rmse_per_channel']} UKF RMSE={ukf_m['rmse_per_channel']}")
        ekf_metrics.append({'id': sid, **ekf_m})
        ukf_metrics.append({'id': sid, **ukf_m})
    pd.DataFrame(ekf_metrics).to_csv(os.path.join(out_dir, 'ekf_metrics_summary.csv'), index=False)
    pd.DataFrame(ukf_metrics).to_csv(os.path.join(out_dir, 'ukf_metrics_summary.csv'), index=False)

    # Prepare data for GRU: split train/val/test
    random.shuffle(sample_ids)
    n_train = int(0.7 * len(sample_ids))
    n_val = int(0.15 * len(sample_ids))
    train_ids = sample_ids[:n_train]; val_ids = sample_ids[n_train:n_train + n_val]; test_ids = sample_ids[n_train + n_val:]
    print(f"\nSplit: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    train_ds = EEGDataset('', out_dir, train_ids)
    val_ds = EEGDataset('', out_dir, val_ids)
    test_ds = EEGDataset('', out_dir, test_ids)

    batch_size = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_pad)

    sample0 = train_ds[0][0].numpy()
    n_channels = sample0.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training GRU denoiser... device:", device)
    model = train_gru(train_loader, val_loader, n_channels, epochs=12, lr=3e-3, device=device)

    # Evaluate GRU on test set (first 5 samples)
    print("\nEvaluating GRU on test set (first 5 samples)...")
    model.eval()
    test_metrics = []
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(test_loader):
            if i >= 5:
                break
            noisy = noisy.to(device); clean = clean.to(device)
            pred = model(noisy).cpu().numpy()[0].T
            noisy_np = noisy.cpu().numpy()[0].T
            clean_np = clean.cpu().numpy()[0].T
            metrics = compute_metrics(pred, clean_np)
            metrics_noisy = compute_metrics(noisy_np, clean_np)
            print(f"Test sample {i}: GRU RMSE {metrics['rmse_per_channel']}, Corr {metrics['corr_per_channel']}")
            print(f"           Noisy RMSE {metrics_noisy['rmse_per_channel']}, Corr {metrics_noisy['corr_per_channel']}")
            test_metrics.append({'sample_index': i, 'gru_rmse': metrics['rmse_per_channel'], 'noisy_rmse': metrics_noisy['rmse_per_channel']})
    pd.DataFrame(test_metrics).to_csv(os.path.join(out_dir, 'gru_test_metrics.csv'), index=False)
    print("\nDone. Dataset and baselines saved in", out_dir)

if __name__ == '__main__':
    main()
