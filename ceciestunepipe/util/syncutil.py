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

logger = logging.getLogger('ceciestunepipe.util.syncutil')

def square_to_edges(x: np.array) -> np.array:
        
        dig = np.squeeze(x)
        diff_dig = np.diff(x)

        rising = np.where(diff_dig > 0)[0]
        falling = np.where(diff_dig < 0)[0]

        ttl_frames = np.concatenate((rising, falling))
        ttl_states = np.array([1] * len(rising) + [-1] * len(falling))
        sort_idxs = np.argsort(ttl_frames)

        return ttl_frames[sort_idxs], ttl_states[sort_idxs]

def quick_ttl_threshold(x:np.array) -> np.float:
    # assumes values oscilate roughly between two points (no outlier noise)
    thresh = np.min(x) + np.ptp(x)/2
    return thresh