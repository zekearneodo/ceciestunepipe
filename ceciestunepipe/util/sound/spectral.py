from scipy import signal
import numpy as np

from ceciestunepipe.util.sound import rosa


def make_butter_bandpass(s_f: float, lo_cut:float, hi_cut:float, order: int=5):
    nyq = 0.5 * s_f
    low = lo_cut / nyq
    high = hi_cut / nyq
    b, a = sg.butter(order, [low, high], btype='band')
    return {'b': b, 'a': a}

def apply_butter_bandpass(x: np.array, pars: dict, axis: int=0):
    return sg.filtfilt(pars['b'], pars['a'], x, axis=axis)


def ms_spectrogram(x, s_f, n_window=512, step_ms=1, f_min=300, f_max=7000, cut_off=0.000055):

    # the overlap is the size of the window minus the smples in a msec
    msec_samples = int(s_f * 0.001)
    n_overlap = n_window - msec_samples * step_ms
    sigma = 1 / 200. * s_f

    # Make the spectrogram
    f, t, Sxx = signal.spectrogram(x, s_f,
                                   nperseg=n_window,
                                   noverlap=n_overlap,
                                   window=signal.gaussian(n_window, sigma),
                                   scaling='spectrum')

    if cut_off > 0:
        Sxx[Sxx < np.max((Sxx) * cut_off)] = 1
    
    Sxx[f<f_min, :] = 1

    return f[(f>f_min) & (f<f_max)], t, Sxx[(f>f_min) & (f<f_max)]

def rosa_spectrogram(y, hparams):
    D = rosa._stft(rosa.preemphasis(y,hparams), hparams)
    S = rosa._amp_to_db(np.abs(D)) - hparams['ref_level_db']
    return rosa._normalize(S, hparams)

def inv_spectrogram(spectrogram, hparams):
    '''Converts spectrogram to waveform using librosa'''
    S = rosa._db_to_amp(rosa._denormalize(spectrogram, hparams) + hparams['ref_level_db'])  # Convert back to linear
    return rosa.inv_preemphasis(rosa._griffin_lim(S ** hparams['power'], hparams), hparams)          # Reconstruct phase

    