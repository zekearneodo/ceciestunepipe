import os
import sys
import glob
import logging
import datetime
import parse
import shutil
import copy
import numpy as np
import pandas as pd
import warnings

from scipy.io import wavfile

from ceciestunepipe.util import syncutil as su

logger = logging.getLogger('ceciestunepipe.util.wavutil')


def save_wav(stream: np.array, s_f: np.float, wav_path: str):
    wav_s_f = int(round(s_f/1000))*1000
    logger.info('sampling rate {}'.format(wav_s_f))
    
    # write to wav
    logger.info('saving {}-shaped array as wav in {}'.format(stream.shape, wav_path))
    
    os.makedirs(os.path.split(wav_path)[0], exist_ok=True, mode=0o777)
    wavfile.write(wav_path, wav_s_f, stream.T)
    
    ## also save as numpy
    npy_path = wav_path.split('.')[0] + '.npy'
    logger.info('saving {}-shaped array as npy in {}'.format(stream.shape, npy_path))
    np.save(npy_path, stream.T)
    
    return wav_s_f

def read_wav_chan(wav_path: str, chan_id: int=0) -> tuple:
    s_f, x = wavfile.read(wav_path, mmap=True)
    if x.ndim==1:
        if chan_id > 0:
            raise ValueError('Wave file has only one channel, asking for channel {}'.format(chan_id))
        x = x.reshape(-1, 1)
    
    return s_f, x[:, chan_id]

def wav_to_syn(file_path:str, chan=0):
    # read the channel from the wave
    # turn the square into a 'digital square' (e.g set a threshold)
    s_f, x = read_wav_chan(file_path, chan_id=chan)
    ttl_thresh = su.quick_ttl_threshold(x)
    
    # make a short 'digital' x
    x_dig = np.zeros_like(x, dtype=np.int8)
    x_dig[x > ttl_thresh] = 1
    
    # get the edges of it
    ttl_events = su.square_to_edges(x_dig)
    ttl_arr = np.vstack(list(ttl_events[:]))
    
    return s_f, x_dig, ttl_arr