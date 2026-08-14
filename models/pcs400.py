# Reference: https://github.com/RoyChao19477/SEMamba/models/pcs400

import numpy as np

# PCS400 parameters
PCS400 = np.ones(201)
PCS400[0:3] = 1
PCS400[3:5] = 1.070175439
PCS400[5:8] = 1.182456140
PCS400[8:10] = 1.287719298
PCS400[10:110] = 1.4       # Pre Set
PCS400[110:130] = 1.322807018
PCS400[130:160] = 1.238596491
PCS400[160:190] = 1.161403509
PCS400[190:202] = 1.077192982

maxv = np.iinfo(np.int16).max
N_FFT = 400
HOP_LENGTH = 100
PCS_WINDOW = np.hamming(N_FFT)

def Sp_and_phase(signal):
    signal_length = signal.shape[0]
    y_pad = np.pad(signal, (0, N_FFT // 2))
    centered = np.pad(y_pad, (N_FFT // 2, N_FFT // 2))
    frames = np.lib.stride_tricks.sliding_window_view(centered, N_FFT)[::HOP_LENGTH]
    F = np.fft.rfft(frames * PCS_WINDOW, n=N_FFT, axis=-1).T
    Lp = PCS400 * np.transpose(np.log1p(np.abs(F)), (1, 0))
    phase = np.angle(F)

    NLp = np.transpose(Lp, (1, 0))

    return NLp, phase, signal_length


def SP_to_wav(mag, phase, signal_length):
    mag = np.expm1(mag)
    Rec = np.multiply(mag, np.exp(1j*phase))
    frames = np.fft.irfft(Rec.T, n=N_FFT, axis=-1) * PCS_WINDOW
    output_length = N_FFT + HOP_LENGTH * (frames.shape[0] - 1)
    result = np.zeros(output_length, dtype=frames.dtype)
    window_norm = np.zeros(output_length, dtype=frames.dtype)
    window_square = PCS_WINDOW ** 2
    for frame_index, frame in enumerate(frames):
        start = frame_index * HOP_LENGTH
        result[start:start + N_FFT] += frame
        window_norm[start:start + N_FFT] += window_square
    valid = window_norm > np.finfo(window_norm.dtype).tiny
    result[valid] /= window_norm[valid]
    start = N_FFT // 2
    result = result[start:start + signal_length]
    if result.shape[0] < signal_length:
        result = np.pad(result, (0, signal_length - result.shape[0]))
    return result

def cal_pcs(signal_wav):
    signal_wav = np.asarray(signal_wav)
    input_dtype = signal_wav.dtype
    noisy_LP, Nphase, signal_length = Sp_and_phase(signal_wav.squeeze())
    enhanced_wav = SP_to_wav(noisy_LP, Nphase, signal_length)
    peak = np.max(np.abs(enhanced_wav))
    if peak > 0.0 and np.isfinite(peak):
        enhanced_wav = enhanced_wav / peak
    else:
        enhanced_wav = np.zeros_like(enhanced_wav)

    return enhanced_wav.astype(input_dtype, copy=False)
