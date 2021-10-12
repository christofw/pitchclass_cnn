"""
Module: LibFMP.C8.C8S2_F0
Author: Sebastian Rosenzweig, Meinard Müller
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
import librosa
from scipy import ndimage, linalg
from numba import jit
import matplotlib.pyplot as plt

import LibFMP.B


def hz_to_cents(F, F_ref=55.0):
    """Converts frequency in Hz to cents

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        F: Frequency value in Hz
        F_ref: Reference frequency in Hz

    Returns:
        Frequency in cents
    """
    F_cent = 1200*np.log2(F/F_ref)
    return F_cent

def cents_to_hz(F_cent, F_ref=55.0):
    """Converts frequency in cents to Hz

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        F: Frequency value in cents
        F_ref: Reference frequency in Hz

    Returns:
        Frequency in Hz
    """
    F = F_ref * 2 ** (F_cent/1200)
    return F

def sonify_trajectory_with_sinusoid(traj, audio_len, Fs=22050, amplitude=0.3, smooth_len=11):
    """Sonification of trajectory with sinusoidal

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        traj: F0 trajectory (time in seconds, frequency in Hz)
        audio_len_samples: Desired audio length in samples
        Fs: Sampling rate
        sine_len: Length of sinusoidal components in sample (hop size)
        smooth_len: Length of amplitude smoothing filter

    Returns:
        x_soni: Sonification
    """
    # unit confidence if not specified
    if traj.shape[1] < 3:
        confidence = np.zeros(traj.shape[0])
        confidence[traj[:,1] > 0] = amplitude
    else:
        confidence = traj[:, 2]

    # initialize
    x_soni = np.zeros(audio_len)
    amplitude_mod = np.zeros(audio_len)

    # Computation of hop size
    #sine_len = int(2 ** np.round(np.log(traj[1, 0]*Fs)/np.log(2)))
    sine_len = int(traj[1, 0]*Fs)

    t = np.arange(0, sine_len)/Fs
    phase = 0

    # loop over all F0 values, insure continuous phase
    for idx in np.arange(0, traj.shape[0]):
        cur_f = traj[idx, 1]
        cur_amp = confidence[idx]

        if cur_f == 0:
            phase = 0
            continue

        cur_soni = np.sin(2*np.pi*(cur_f*t+phase))
        diff = np.maximum(0, (idx+1)*sine_len - len(x_soni))
        if diff > 0:
            x_soni[idx * sine_len:(idx + 1) * sine_len - diff] = cur_soni[:-diff]
            amplitude_mod[idx * sine_len:(idx + 1) * sine_len - diff] = cur_amp
        else:
            x_soni[idx*sine_len:(idx+1)*sine_len-diff] = cur_soni
            amplitude_mod[idx*sine_len:(idx+1)*sine_len-diff] = cur_amp

        phase += cur_f*sine_len/Fs
        phase -= 2*np.round(phase/2)

    # filter amplitudes to avoid transients
    amplitude_mod = np.convolve(amplitude_mod, np.hanning(smooth_len)/np.sum(np.hanning(smooth_len)), 'same')
    x_soni = x_soni * amplitude_mod
    return x_soni

def visualize_salience_traj_constraints(Z, T_coef, F_coef_cents, F_ref=55.0, colorbar=True, cmap='gray_r',
                                        figsize=(7,4), traj=None, constraint_region=None, ax=None):
    """Visualize salience representation with optional F0-trajectory and constraint regions

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        Z: Salience representation
        T_coef: Time axis
        F_coef_cents: Frequency axis in cents
        F_ref: Reference frequency
        colorbar (bool): Show or hide colorbar
        contour: F0 contour, time in seconds in 1st column, frequency in Hz in 2nd column
        constraint_region: Constraint regions, row-format: (t_start_sec, t_end_sec, f_start_hz, f_end,hz)
        ax: Handle to existing axis
    Returns:
        fig: Handle to figure
        ax: Handle to cent axis
        ax_f: Handle to frequency axis
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    sal = ax.imshow(Z, extent=[T_coef[0], T_coef[-1], F_coef_cents[0], F_coef_cents[-1]],
                    cmap=cmap, origin='lower', aspect='auto')

    y_ticklabels_left = np.arange(F_coef_cents[0],F_coef_cents[-1]+1,1200)
    ax.set_yticks(y_ticklabels_left);
    ax.set_yticklabels(y_ticklabels_left);
    ax.set_ylabel('Frequency (Cents)');

    plt.colorbar(sal, ax=ax, pad=0.1)

    ax_f = ax.twinx();  # instantiate a second axes that shares the same y-axis
    ax_f.set_yticks(y_ticklabels_left-F_coef_cents[0]);
    y_ticklabels_right = cents_to_hz(y_ticklabels_left, F_ref).astype(int)
    ax_f.set_yticklabels(y_ticklabels_right);
    ax_f.set_ylabel('Frequency (Hz)');

    # plot contour
    if traj is not None:
        traj_plot = traj[traj[:, 1] > 0, :]
        traj_plot[:, 1] = hz_to_cents(traj_plot[:, 1], F_ref)
        ax.plot(traj_plot[:, 0], traj_plot[:, 1], color='r', markersize=4, marker='.', linestyle='');

    # plot constraint regions
    if constraint_region is not None:
        for row in constraint_region:
            t_start = row[0]  # sec
            t_end = row[1]  # sec
            f_start = row[2]  # Hz
            f_end = row[3]  # Hz
            ax.add_patch(matplotlib.patches.Rectangle((t_start,
                    hz_to_cents(f_start, F_ref)), width=t_end-t_start,
                    height=hz_to_cents(f_end, F_ref)-hz_to_cents(f_start, F_ref),
                    fill=False, edgecolor='k',linewidth=3, zorder=2))

    ax.set_xlabel('Time (seconds)');

    if fig is not None:
        plt.tight_layout()

    return fig, ax, ax_f

#@jit(nopython=True)
def define_transition_matrix(B, tol=0, score_low=0.01, score_high=1):
    """Generate transition matrix

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        B: Number of bins
        tol: Tolerance parameter for transition matrix
        score_low, score_high: Scores for transition matrix

    Returns:
        T: Transition matrix
    """
    col = np.ones((B,)) * score_low
    col[0:tol+1] = np.ones((tol+1,)) * score_high
    T = linalg.toeplitz(col)
    return T

@jit(nopython=True)
def compute_trajectory_DP(Z, T):
    """Trajectory tracking using dynamic programming

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        salience: Salience representation
        T: Transisition matrix

    Returns:
        eta_DP: Trajectory indices
    """
    B, N = Z.shape
    eps_machine = np.finfo(np.float32).eps
    Z_log = np.log(Z + eps_machine)
    T_log = np.log(T + eps_machine)

    E = np.zeros((B, N))
    D = np.zeros((B, N))
    D[:, 0] = Z_log[:, 0]

    for n in np.arange(1, N):
        for b in np.arange(0, B):
            D[b, n] = np.max(T_log[b, :] + D[:, n-1]) + Z_log[b, n]
            E[b, n-1] = np.argmax(T_log[b, :] + D[:, n-1])

    # backtracking
    eta_DP = np.zeros(N)
    eta_DP[N-1] = int(np.argmax(D[:, N-1]))

    for n in np.arange(N-2, -1, -1):
        eta_DP[n] = E[int(eta_DP[n+1]), n]

    return eta_DP.astype(np.int64)

def convert_ann_to_constraint_region(ann, tol_freq_cents=300):
    """Convert score annotations to constraint regions

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        ann: score annotations [[start_time, end_time, MIDI_pitch], ...
        tol_freq_cents: Tolerance in pitch directions specified in cents

    Returns:
        eta_DP: Trajectory indices
    """
    tol_pitch = tol_freq_cents/100
    freq_lower = 2 ** ((ann_score[:, 2] - tol_pitch - 69)/12) * 440
    freq_upper = 2 ** ((ann_score[:, 2] + tol_pitch - 69)/12) * 440
    constraint_region = np.concatenate((ann_score[:,0:2],
                                        freq_lower.reshape(-1,1),
                                        freq_upper.reshape(-1,1)), axis=1)
    return constraint_region

#@jit(nopython=True)
def compute_trajectory_CR(Z, T_coef, F_coef_hertz, constraint_region=None,
                          tol=5, score_low=0.01, score_high=1):
    """Trajectory tracking with constraint regions

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        Z: Salience representation
        T_coef: Time axis
        F_coef_hertz: Frequency axis in Hz
        constraint_region: Constraint regions, row-format: (t_start_sec, t_end_sec, f_start_hz, f_end,hz)
        tol: Tolerance parameter for transition matrix
        score_low, score_high: Scores for transition matrix

    Returns:
        eta: Trajectory indices, unvoiced frames are indicated with -1
    """
    # do tracking within every constraint region
    if constraint_region is not None:
        # initialize contour, unvoiced frames are indicated with -1
        eta = np.full(len(T_coef), -1)

        for row_idx in range(constraint_region.shape[0]):
            t_start = constraint_region[row_idx, 0]  # sec
            t_end = constraint_region[row_idx, 1]  # sec
            f_start = constraint_region[row_idx, 2]  # Hz
            f_end = constraint_region[row_idx, 3]  # Hz

            # convert start/end values to indices
            t_start_idx = np.argmin(np.abs(T_coef - t_start))
            t_end_idx = np.argmin(np.abs(T_coef - t_end))
            f_start_idx = np.argmin(np.abs(F_coef_hertz - f_start))
            f_end_idx = np.argmin(np.abs(F_coef_hertz - f_end))

            # track in salience part
            cur_Z = Z[f_start_idx:f_end_idx+1, t_start_idx:t_end_idx+1]
            T = define_transition_matrix(cur_Z.shape[0], tol=tol,
                                         score_low=score_low, score_high=score_high)
            cur_eta = compute_trajectory_DP(cur_Z, T)

            # fill contour
            eta[t_start_idx:t_end_idx+1] = f_start_idx + cur_eta
    else:
        T = define_transition_matrix(Z.shape[0], tol=tol, score=score)
        eta = compute_trajectory_DP(Z, T)

    return eta

def compute_traj_from_audio(x, Fs=22050, N=1024, H=128, R=10, F_min=55, F_max=1760,
                            num_harm=10, freq_smooth_len=11, alpha=0.9, gamma=0,
                            constraint_region=None, tol=5, score_low=0.01, score_high=1):
    """Compute F0 contour from audio signal

    Notebook: C8/C8S2_FundFreqTracking.ipynb

    Args:
        x: Audio signal
        Fs: Sampling frequency
        N: Window length in samples
        H: Hopsize in samples
        R: Frequency resolution in cents
        F_min: Lower frequency bound (reference frequency)
        F_max: Upper frequency bound
        num_harm: Number of harmonics
        freq_smooth_len: Filter length for vertical smoothing
        alpha: Weighting parameter for harmonics
        gamma: Logarithmic compression factor
        constraint_regions: Constraint regions, row-format: (t_start_sec, t_end_sec, f_start_hz, f_end,hz)
        tol: Tolerance parameter for transition matrix
        score_low, score_high: Scores for transition matrix

    Returns:
        contour: F0 contour, time in seconds in 1st column, frequency in Hz in 2nd column
    """
    Z, F_coef_hertz, F_coef_cents = LibFMP.C8.compute_salience_rep(x, Fs, N=N, H=H, R=R,
            F_min=F_min, F_max=F_max, num_harm=num_harm, freq_smooth_len=freq_smooth_len, alpha=alpha, gamma=gamma)

    T_coef =  (np.arange(Z.shape[1]) * H) / Fs
    index_CR = compute_trajectory_CR(Z, T_coef, F_coef_hertz, constraint_region,
                                     tol=tol, score_low=score_low, score_high=score_high)

    traj = np.hstack((T_coef.reshape(-1, 1), F_coef_hertz[index_CR].reshape(-1, 1)))
    traj[index_CR==-1, 1] = 0
    return traj, Z, T_coef, F_coef_hertz, F_coef_cents

def convert_trajectory_to_mask_bin(traj, F_coef, n_harmonics=1, tol_bin=0):
    """Computes binary mask from F0 trajectory

    Notebook: C8/C8S2_MelodyExtractSep.ipynb

    Args:
        traj: F0 trajectory (time in seconds in 1st column, frequency in Hz in 2nd column)
        F_coef: Frequency axis
        n_harmonics: Number of harmonics
        tol_bin: Tolerance in frequency bins

    Returns:
        mask: Binary mask
    """
    # Compute STFT bin for trajectory
    traj_bin = np.argmin(np.abs(F_coef[:, None] - traj[:, 1][None, :]), axis=0)

    K = len(F_coef)
    N = traj.shape[0]
    max_idx_harm = np.max([K, np.max(traj_bin)*n_harmonics])
    mask_pad = np.zeros((max_idx_harm.astype(int)+1, N))

    for h in range(n_harmonics):
        mask_pad[traj_bin*h, np.arange(N)] = 1
    mask = mask_pad[1:K+1, :]

    if tol_bin > 0:
        smooth_len = 2*tol_bin + 1
        mask = ndimage.filters.maximum_filter1d(mask, smooth_len, axis=0, mode='constant', cval=0, origin=0)

    return mask

def convert_trajectory_to_mask_cent(traj, F_coef, n_harmonics=1, tol_cent=0):
    """Computes binary mask from F0 trajectory

    Notebook: C8/C8S2_MelodyExtractSep.ipynb

    Args:
        traj: F0 trajectory (time in seconds in 1st column, frequency in Hz in 2nd column)
        F_coef: Frequency axis
        n_harmonics: Number of harmonics
        tol_cent: Tolerance in cents

    Returns:
        mask: Binary mask
    """
    K = len(F_coef)
    N = traj.shape[0]
    mask = np.zeros((K,N))

    freq_res = F_coef[1] - F_coef[0]
    tol_factor = np.power(2, tol_cent/1200)
    F_coef_upper = F_coef * tol_factor
    F_coef_lower = F_coef / tol_factor
    F_coef_upper_bin = (np.ceil(F_coef_upper / freq_res)).astype(int)
    F_coef_upper_bin[F_coef_upper_bin > K-1] = K-1
    F_coef_lower_bin = (np.floor(F_coef_lower / freq_res)).astype(int)

    for n in range(N):
        for h in range(n_harmonics):
            freq = traj[n, 1] * (1 + h)
            freq_bin = np.round(freq / freq_res).astype(int)
            if freq_bin < K:
                idx_upper = F_coef_upper_bin[freq_bin]
                idx_lower = F_coef_lower_bin[freq_bin]
                mask[idx_lower:idx_upper+1, n] = 1
    return mask

def separate_melody_accompaniment(x, Fs, N, H, traj, n_harmonics=10, tol_cent=50):
    """F0-based melody-accompaniement separation

    Notebook: C8/C8S2_MelodyExtractSep.ipynb

    Args:
        x: Audio signal
        Fs: Sampling frequency
        N: Window size in samples
        H: Hopsize in samples
        traj: F0 traj (time in seconds in 1st column, frequency in Hz in 2nd column)
        n_harmonics: Number of harmonics
        tol_cent: Tolerance in cents

    Returns:
        x_mel: Reconstructed audio signal for melody
        x_acc: Reconstructed audio signal for accompaniement
    """
    # Compute STFT
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, pad_mode='constant')
    Fs_feature = Fs/H
    T_coef = np.arange(X.shape[1]) / Fs_feature
    freq_res = Fs / N
    F_coef = np.arange(X.shape[0]) * freq_res

    # Adjust trajectory
    traj_X_values = interp1d(traj[:,0], traj[:,1], kind='nearest', fill_value='extrapolate')(T_coef)
    traj_X = np.hstack((T_coef[:,None], traj_X_values[:,None,]))

    # Compute binary masks
    mask_mel = convert_trajectory_to_mask_cent(traj_X, F_coef, n_harmonics=n_harmonics, tol_cent=tol_cent)
    mask_acc = np.ones(mask_mel.shape) - mask_mel

    # Compute masked STFTs
    X_mel = X * mask_mel
    X_acc = X * mask_acc

    # Reconstruct signals
    x_mel = librosa.istft(X_mel, hop_length=H, win_length=N, window='hann', center=True, length=x.size)
    x_acc = librosa.istft(X_acc, hop_length=H, win_length=N, window='hann', center=True, length=x.size)

    return x_mel, x_acc
