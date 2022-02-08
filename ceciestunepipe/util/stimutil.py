import numpy as np
import pandas as pd
import warnings
import logging
import json
import os
from typing import Union

from ceciestunepipe.util.sound import spectral as sp

logger = logging.getLogger("ceciestunepipe.util.stimutil")


def find_first_peak(x: np.array, thresh_factor: np.float = 0.3) -> np.int:
    """ Finds first peak of a waveform. It works best when looking for the onset of a sine wave.

    Args:
        x (np.array): (n, ) array of values
        thresh_factor (np.float, optional): Threshold as a factor of the ampitude of the waveform. Defaults to 0.3.

    Returns:
        np.int: Index of the found peak
    """
    x = x - np.mean(x)
    thresh = np.max(x) * thresh_factor
    # find the peaks naively
    a = x[1:-1] - x[2:]
    b = x[1:-1] - x[:-2]
    c = x[1:-1]
    max_pos = np.where((a >= 0) & (b > 0) & (c > thresh))[0] + 1
    return max_pos[0]


def round_to(x, n: int=100):
    x_r = round(x/n) * n
    return x_r


def find_wav_onset(dset, chan, stamp, tr_df):
    [start, end] = get_trial_bounds(stamp, tr_df)
    # logger.debug('Finding onset around {0}-{1} for {2}'.format(start, end, stamp))
    trial_frame = h5t.load_table_slice(dset, np.arange(start, end), [chan])
    onset_in_trial = find_first_peak(trial_frame)
    return start + onset_in_trial


def get_sine_freq(x, s_f, samples=1024):
    f, t, s = sp.pretty_spectrogram(x[:samples].astype(np.float), s_f,
                                    log=False,
                                    fft_size=samples,
                                    step_size=samples,
                                    window=('tukey', 0.25))
    # s should be just one slice
    # get the peak frequency
    f_0 = f[np.argmax(s[:, 0])]

    return f_0


def get_sine(x, s_f, trial_bound):
    # onset ref to the beginning of the rec
    onset = find_first_peak(x[trial_bound[0]:trial_bound[1]]) + trial_bound[0]
    sin_chunk = x[onset: trial_bound[1]]

    f_0 = get_sine_freq(sin_chunk, s_f)
    #logger.debug('f0 {}'.format(f_0))

    if (np.isfinite(f_0) and f_0 > 0):
        # correct the onset with the 1/4 wave
        wave_samples = float(s_f) / f_0
        samples_correction = int(wave_samples * 0.25)
        # print(samples_correction)
    else:
        msg = 'Invalid f_0 around {}'.format(trial_bound)
        warnings.warn(msg, RuntimeWarning)
        samples_correction = 0

    return onset - samples_correction, onset, f_0


def get_trials_from_ttl(x: np.array, on_signal: int = 1) -> np.array:
    """ Make a pandas dataframe with the onset/offset of each 'trial' as defined by the onset/offset of a TTL event

    Args:
        x (np.array): (2, n), Array with digital events (sample, event=+/-1), 1 row per event
        on_signal (int, optional): Whether the onset is signaled by hi(1) or lo(-1). Defaults to 1.

    Returns:
        x (np.array): (2, m), Array with onset/offset pairs. 1 row per interval. m=n/2 if all intervals were caught.
    """
    # filter:
    # - check for corruption (two highs/two lows)
    skipped = np.sum(np.diff(x[::2, 1]))

    # - in the end, every offset has to have an onset
    # - every interval (off - on) is positive
    if(skipped > 0):
        raise RuntimeError(
            'One or more consecutive events in the same direction; the events ttl file is corrupted or missed some triggers')

    # - first offset after first onset
    if x[0, 0] == -on_signal:
        warnings.warn('The first event was an offset, skipping that')
        x = x[1:, :]

    # - last offset after last onset
    if x[-1, 0] == on_signal:
        warnings.warn('The last event was an onset, skipping that')
        x = x[:-1, :]

    # get all intervals:
    # this should be easy now, just all interleaved
    onof = np.vstack([x[::2, 0], x[1::2, 0]])

    # double check that all intervals are positive
    intervals = np.diff(onof, axis=1)
    if np.any(intervals < 0):
        raise RuntimeError(
            'One or more on intervals is negative; the events ttl file is corrupted or missed some triggers. Have you checked the sign of the trigger?')

    return onof


def get_trials_info(onof_arr: np.array, tag_stream: np.array, tag_s_f: np.float):
    # for every onof, look for the onset and offset of the sinewave within, and get the frequency
    logger.info(
        'Looking for precise star/end of {} stimulus trials'.format(onof_arr.size/2))

    logger.info('forward pass (onsets)')
    all_starts = np.array(list(map(lambda tb: get_sine(tag_stream.flatten(), tag_s_f, tb), onof_arr.T)),
                          dtype=np.int64)

    # reverse to find the offsets
    logger.info('reverse pass (offsets)')
    ofon_arr = tag_stream.size - onof_arr[::-1, ::-1]
    tag_stream_rev = tag_stream.flatten()[::-1]
    all_ends = np.array(list(map(lambda tb: get_sine(tag_stream_rev, tag_s_f, tb), ofon_arr.T)),
                        dtype=np.int64)

    # re_reverse
    all_ends[:, :2] = tag_stream.size - all_ends[:, :2]
    all_ends = all_ends[::-1]

    return all_starts, all_ends


def stim_dict_to_pd(tag_dict: dict) -> pd.DataFrame:
    # make a dataframe with tag_freq, stim
    tag_pd = pd.DataFrame({'tag_freq': list(tag_dict.values()), 
                 'stim_name': list(tag_dict.keys())})
    return tag_pd

def get_trials_pd(trial_ttl: np.array, trial_stream:np.array, stim_s_f, on_signal: int=1, tag_chan: int=1):
    
    # get the on/of
    onof = get_trials_from_ttl(trial_ttl, on_signal=on_signal)
    # get the precise on/ofs and the sine frequencies
    all_starts, all_ends =  get_trials_info(onof, trial_stream[:, tag_chan], stim_s_f)
    
    # check consistency
    match_freq = (all_starts[:, 2]//10 == all_ends[:, 2]//10)
    if not all(match_freq):
        raise RuntimeError("Frequency mismatch between found in onset and offset")
    # make into a pd
    trials_pd = pd.DataFrame(np.vstack([all_starts[:, 0], all_ends[:, 1] + 1, all_starts[:, 2]]).T,
    columns=['start', 'end', 'tag_freq_int'])

    # if given use a dictionary to relate frequency to wav file 

    return trials_pd

def get_trial_stim_names(trial_pd: pd.DataFrame, stim_tags_dict: dict) -> pd.DataFrame:
    # round the frequency tag of the trials_pd to 100 hertz
    trial_pd['tag_freq'] = trial_pd['tag_freq_int'].apply(lambda x: round_to(x, 100))
    
    # fill in the stim_file for each trial in trial_pd with the one with the matching frequencies
    stim_tag_pd = stim_dict_to_pd(stim_tags_dict)
    trial_tagged_pd = trial_pd.merge(stim_tag_pd, how='left', on='tag_freq')

    return trial_tagged_pd
    
def get_wav_stims(trials_path: str, stim_s_f: float, event_name: str='wav', on_signal: int=1, tag_chan: int=1, 
tag_dict: dict=None):

    # get the streams
    ttl_ev_name = event_name + '_ttl'
    stream_name = event_name + '_stim'

    npy_stim_path = os.path.join(trials_path, ttl_ev_name + '_evt.npy')
    stream_stim_path = os.path.join(trials_path, stream_name + '.npy')

    trial_ttl = np.load(npy_stim_path)
    trial_stream = np.load(stream_stim_path, mmap_mode='r')

    trial_pd = get_trials_pd(trial_ttl, trial_stream, stim_s_f, on_signal=on_signal, tag_chan=tag_chan)

    # if there was a tag_dict, use it to match stim names in the trial pd
    if tag_dict:
        logger.info('Using entered stim_tags_dict to fill in stimuli in the trial DataFrame')
        trial_pd = get_trial_stim_names(trial_pd, tag_dict)
    else:
        trial_pd['stim_name'] = None

    return trial_pd


