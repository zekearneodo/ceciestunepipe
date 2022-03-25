import more_itertools as mit
import peakutils
import warnings
import traceback
import numpy as np
import pandas as pd
import pickle
import logging
import os
import glob
import sys
import traceback

from scipy.io import wavfile
from tqdm.auto import tqdm

from ceciestunepipe.util import fileutil as fu
from ceciestunepipe.util.sound import spectral as sp
from ceciestunepipe.util.sound import temporal as st


logger = logging.getLogger('ceciestunepipe.util.sound.boutsearch')

def read_wav_chan(wav_path: str, chan_id: int=0) -> tuple:
    s_f, x = wavfile.read(wav_path, mmap=True)
    return s_f, x[:, chan_id]

def sess_file_id(f_path):
    n = int(os.path.split(f_path)[1].split('-')[-1].split('.wav')[0])
    return n

def sample_rate_from_wav(wav_path):
    sample_rate, x = wavfile.read(wav_path)
    return sample_rate

default_hparams = { # default parameters work well for starling
    # spectrogram
    'num_freq':1024, #1024# how many channels to use in a spectrogram #
    'preemphasis':0.97, 
    'frame_shift_ms':5, # step size for fft
    'frame_length_ms':10, #128 # frame length for fft FRAME SAMPLES < NUM_FREQ!!!
    'min_level_db':-55, # minimum threshold db for computing spe 
    'ref_level_db':110, # reference db for computing spec
    'sample_rate':None, # sample rate of your data
    
    # spectrograms
    'mel_filter': False, # should a mel filter be used?
    'num_mels':1024, # how many channels to use in the mel-spectrogram
    'fmin': 500, # low frequency cutoff for mel filter
    'fmax': 12000, # high frequency cutoff for mel filter
    
    # spectrogram inversion
    'max_iters':200,
    'griffin_lim_iters':20,
    'power':1.5,

    # Added for the searching
    'read_wav_fun': read_wav_chan, # function for loading the wav_like_stream (has to returns fs, ndarray)
    'file_order_fun': sess_file_id, # function for extracting the file id within the session
    'min_segment': 30, # Minimum length of supra_threshold to consider a 'syllable' (ms)
    'min_silence': 2000, # Minmum distance between groups of syllables to consider separate bouts (ms)
    'min_bout': 5000, # min bout duration (ms)
    'peak_thresh_rms': 0.55, # threshold (rms) for peak acceptance,
    'thresh_rms': 0.25, # threshold for detection of syllables
    'mean_syl_rms_thresh': 0.3, #threshold for acceptance of mean rms across the syllable (relative to rms of the file)
    'max_bout': 120000, #exclude bouts too long
    'l_p_r_thresh': 100, # threshold for n of len_ms/peaks (typycally about 2-3 syllable spans
    
    'waveform_edges': 1000, #get number of ms before and after the edges of the bout for the waveform sample
    
    'bout_auto_file': 'bout_auto.pickle', # extension for saving the auto found files
    'bout_curated_file': 'bout_checked.pickle', #extension for manually curated files (coming soon)
    }


def gimmepower(x, hparams):
    s = sp.rosa_spectrogram(x.flatten(), hparams)
    f = np.arange(hparams['num_freq'])/hparams['num_freq']*0.5*hparams['sample_rate']
    s = s[(f>hparams['fmin']) & (f<hparams['fmax']), :]
    f = f[(f>hparams['fmin']) & (f<hparams['fmax'])]
    p = s.sum(axis=0)
    return p, f, s

def get_on_segments(x, thresh=0, min_segment=20, pk_thresh=0, mean_rms_thresh=0):
    on = np.where(x > thresh)[0]
    on_segments = [list(group) for group in mit.consecutive_groups(on)]
    logger.debug('On segments {}'.format(len(on_segments)))
    if len(on_segments) > 0:
        hi_segments = np.vstack([np.array([o[0], o[-1]]) for o in on_segments \
                                 if ((np.max(x[o]) > pk_thresh) and \
                                     (st.rms(x[o]) > mean_rms_thresh))])
    else:
        hi_segments = np.array([])
    if len(hi_segments) > 0:
        long_enough_segments = hi_segments[(np.diff(hi_segments) >= min_segment).flatten(), :]
    else:
        long_enough_segments = np.array([])
        
    logger.debug('good segments shape {}'.format(long_enough_segments.shape))
    return long_enough_segments

def merge_near_segments(on_segs, min_silence=200):
    # merge all segments distant less than min_silence
    # need to have at least two bouts
    if on_segs.shape[0] < 2:
        logger.debug('Less than two zero segments, nothing to possibly merge')
        long_segments = on_segs
    else:
        of_on = on_segs.flatten()[1:]
        silence = np.diff(of_on)[::2]
        long_silence = np.where(silence > min_silence)[0]
        if(long_silence.size == 0):
            logger.debug('No long silences found, all is one big bout')
        of_keep = np.append((on_segs[long_silence, 1]), on_segs[-1, 1])
        on_keep = np.append(on_segs[0,0], on_segs[long_silence + 1, 0])
        long_segments = np.vstack([on_keep, of_keep]).T
    return long_segments

def get_the_bouts(x, spec_par_rosa, loaded_p=None):
    #   
    if loaded_p is not None:
        p = loaded_p
        logger.debug('loaded p with shape {}'.format(loaded_p.shape))
    else:
        logger.debug('Computing power')
        p, _, _ = gimmepower(x, spec_par_rosa)
    
    logger.debug('Finding on segments')
    threshold = spec_par_rosa['thresh_rms'] * st.rms(p)
    pk_threshold = spec_par_rosa['peak_thresh_rms'] * st.rms(p)
    mean_rms_threshold = spec_par_rosa['mean_syl_rms_thresh'] * st.rms(p)
    step_ms = spec_par_rosa['frame_shift_ms']
    min_syl = spec_par_rosa['min_segment'] // step_ms
    min_silence = spec_par_rosa['min_silence'] // step_ms
    min_bout = spec_par_rosa['min_bout'] // step_ms
    max_bout = spec_par_rosa['max_bout'] // step_ms

    syllables = get_on_segments(p, threshold, min_syl, pk_threshold, mean_rms_threshold)
    logger.debug('Found {} syllables'.format(syllables.shape[0]))
    
    logger.debug('Merging segments with silent interval smaller than {} steps'.format(min_silence))
    bouts = merge_near_segments(syllables, min_silence=min_silence)
    logger.debug('Found {} bout candidates'.format(bouts.shape[0]))
    
    if bouts.shape[0] > 0:
        long_enough_bouts = bouts[((np.diff(bouts) >= min_bout) & (np.diff(bouts) < max_bout)).flatten(), :]
        logger.debug('Removed shorter/longer than [{} ;{}], {} candidates left'.format(min_bout, max_bout,
                                                                long_enough_bouts.shape[0]))
    else:
        long_enough_bouts = bouts        
    power_values = [p[x[0]:x[1]] for x in long_enough_bouts]
    
    return long_enough_bouts, power_values, p, syllables

def get_bouts_in_file(file_path, hparams, loaded_p=None):
    # path of the wav_file
    # h_params from the rosa spectrogram plus the parameters:
    #     'read_wav_fun': load_couple, # function for loading the wav_like_stream (has to returns fs, ndarray)
    #     'min_segment': 30, # Minimum length of supra_threshold to consider a 'syllable'
    #     'min_silence': 200, # Minmum distance between groups of syllables to consider separate bouts
    #     'bout_lim': 200, # same as min_dinscance !!! Clean that out!
    #     'min_bout': 250, # min bout duration
    #     'peak_thresh_rms': 2.5, # threshold (rms) for peak acceptance,
    #     'thresh_rms': 1 # threshold for detection of syllables

    # Decide and see if it CAN load the power
    logger.info('Getting bouts for file {}'.format(file_path))
    print('file {}...'.format(file_path))
    sys.stdout.flush()
    #logger.debug('s_f {}'.format(s_f))
    
    # Get the bouts. If loaded_p is none, it will copute it
    try:
        s_f, wav_i = hparams['read_wav_fun'](file_path)
        hparams['sample_rate'] = s_f
        the_bouts, the_p, all_p, all_syl = get_the_bouts(wav_i, hparams, loaded_p=loaded_p)
    
    except Exception as e:
        warnings.warn('Error getting bouts for file {}'.format(file_path))
        logger.info('error {}'.format(e))
        print('error in {}'.format(file_path))
        sys.stdout.flush()
        traceback.print_exception()
        # return empty DataFrame
        the_bouts = np.empty(0)
        wav_i = np.empty(0)
        all_p = np.empty(0)
        

    if the_bouts.size > 0:
        step_ms = hparams['frame_shift_ms']
        pk_dist = hparams['min_segment']
        bout_pd = pd.DataFrame(the_bouts * step_ms, columns=['start_ms', 'end_ms'])
        bout_pd['start_sample'] = bout_pd['start_ms'] * (s_f//1000)
        bout_pd['end_sample'] = bout_pd['end_ms'] * (s_f//1000)
        
        bout_pd['p_step'] = the_p
        # the extrema over the file
        bout_pd['rms_p'] = st.rms(all_p)
        bout_pd['peak_p'] = bout_pd['p_step'].apply(np.max)
        # check whether the peak power is larger than hparams['peak_thresh_rms'] times the rms through the file
        bout_pd['bout_check'] = bout_pd.apply(lambda row: \
                                              (row['peak_p'] > hparams['peak_thresh_rms'] * row['rms_p']), 
                                              axis=1)
        bout_pd['file'] = file_path
        bout_pd['len_ms'] = bout_pd.apply(lambda r: r['end_ms'] - r['start_ms'], axis=1)
        
        syl_pd = pd.DataFrame(all_syl * step_ms, columns=['start_ms', 'end_ms'])
        bout_pd['syl_in'] = bout_pd.apply(lambda r: \
                                          syl_pd[(syl_pd['start_ms'] >= r['start_ms']) & \
                                                 (syl_pd['start_ms'] <= r['end_ms'])].values, 
                                          axis=1)
        bout_pd['n_syl'] = bout_pd['syl_in'].apply(len)
        # get all the peaks larger than the threshold(peak_thresh_rms * rms)
        bout_pd['peaks_p'] = bout_pd.apply(lambda r: peakutils.indexes(r['p_step'], 
                                                                       thres=hparams['peak_thresh_rms']*r['rms_p']/r['p_step'].max(),
                                                                       min_dist=pk_dist//step_ms),
                                           axis=1)
        bout_pd['n_peaks'] = bout_pd['peaks_p'].apply(len)
        bout_pd['l_p_ratio'] = bout_pd.apply(lambda r: np.nan if r['n_peaks']==0 else r['len_ms'] / (r['n_peaks']), axis=1)
        
        try:
            delta = int(hparams['waveform_edges'] * hparams['sample_rate'] * 0.001)
        except KeyError:
            delta = 0

        bout_pd['waveform'] = bout_pd.apply(lambda df: wav_i[df['start_sample'] - delta: df['end_sample'] + delta], axis=1)

    else:
        bout_pd = pd.DataFrame()
    return bout_pd, wav_i, all_p
  

def get_bouts_in_long_file(file_path, hparams, loaded_p=None, chunk_size=90000000):
    # path of the wav_file
    # h_params from the rosa spectrogram plus the parameters:
    #     'read_wav_fun': load_couple, # function for loading the wav_like_stream (has to returns fs, ndarray)
    #     'min_segment': 30, # Minimum length of supra_threshold to consider a 'syllable'
    #     'min_silence': 200, # Minmum distance between groups of syllables to consider separate bouts
    #     'bout_lim': 200, # same as min_dinscance !!! Clean that out!
    #     'min_bout': 250, # min bout duration
    #     'peak_thresh_rms': 2.5, # threshold (rms) for peak acceptance,
    #     'thresh_rms': 1 # threshold for detection of syllables

    # Decide and see if it CAN load the power
    logger.info('Getting bouts for long file {}'.format(file_path))
    #print('tu vieja file {}'.format(file_path))
    sys.stdout.flush()
    #logger.debug('s_f {}'.format(s_f))
    
    # Get the bouts. If loaded_p is none, it will copute it
    s_f, wav_i = hparams['read_wav_fun'](file_path)
    hparams['sample_rate'] = s_f

    n_chunks = int(np.ceil(wav_i.shape[0]/chunk_size))
    wav_chunks = np.array_split(wav_i, n_chunks)
    logger.info('splitting file into {} chunks'.format(n_chunks))

    chunk_start_sample = 0
    bouts_pd_list = []
    for i_chunk, wav_chunk in tqdm(enumerate(wav_chunks), total=n_chunks):
        # get the bouts for a chunk
        # offset the starts to the beginning of the chunk
        # recompute the beginning of the next chunk
        the_bouts, the_p, all_p, all_syl = get_the_bouts(wav_chunk, hparams, loaded_p=loaded_p)
        chunk_offset_ms = int(1000 * chunk_start_sample / s_f)
        
        # make one bouts pd dataframe for the chunk
        if the_bouts.size > 0:
            step_ms = hparams['frame_shift_ms']
            pk_dist = hparams['min_segment']
            bout_pd = pd.DataFrame(the_bouts * step_ms + chunk_offset_ms, columns=['start_ms', 'end_ms'])
            bout_pd['start_sample'] = bout_pd['start_ms'] * (s_f//1000)
            bout_pd['end_sample'] = bout_pd['end_ms'] * (s_f//1000)
            
            bout_pd['p_step'] = the_p
            # the extrema over the file
            bout_pd['rms_p'] = st.rms(all_p)
            bout_pd['peak_p'] = bout_pd['p_step'].apply(np.max)
            # check whether the peak power is larger than hparams['peak_thresh_rms'] times the rms through the file
            bout_pd['bout_check'] = bout_pd.apply(lambda row: \
                                                (row['peak_p'] > hparams['peak_thresh_rms'] * row['rms_p']), 
                                                axis=1)
            bout_pd['file'] = file_path
            bout_pd['len_ms'] = bout_pd.apply(lambda r: r['end_ms'] - r['start_ms'], axis=1)
            
            syl_pd = pd.DataFrame(all_syl * step_ms + + chunk_offset_ms, columns=['start_ms', 'end_ms'])
            bout_pd['syl_in'] = bout_pd.apply(lambda r: \
                                            syl_pd[(syl_pd['start_ms'] >= r['start_ms']) & \
                                                    (syl_pd['start_ms'] <= r['end_ms'])].values, 
                                            axis=1)
            bout_pd['n_syl'] = bout_pd['syl_in'].apply(len)
            # get all the peaks larger than the threshold(peak_thresh_rms * rms)
            bout_pd['peaks_p'] = bout_pd.apply(lambda r: peakutils.indexes(r['p_step'], 
                                                                        thres=hparams['peak_thresh_rms']*r['rms_p']/r['p_step'].max(),
                                                                        min_dist=pk_dist//step_ms),
                                            axis=1)
            bout_pd['n_peaks'] = bout_pd['peaks_p'].apply(len)
            bout_pd['l_p_ratio'] = bout_pd.apply(lambda r: np.nan if r['n_peaks']==0 else r['len_ms'] / (r['n_peaks']), axis=1)
            
            # ### refer the starts, ends to the beginning of the chunk
            # delta_l = -1*delta - chunk_start_sample
            # delta_r = delta - cunk_start_sample
            # bout_pd['waveform'] = bout_pd.apply(lambda df: wav_chunk[df['start_sample'] + delta_l: df['end_sample'] + delta_r], axis=1)

        else:
            bout_pd = pd.DataFrame()
        
        chunk_start_sample += wav_chunk.size
        bouts_pd_list.append(bout_pd)
    
    all_bout_pd = pd.concat(bouts_pd_list)
    all_bout_pd.reset_index(inplace=True, drop=True)
    ###get all the waveforms
    if not all_bout_pd.empty:
        try:
            delta = int(hparams['waveform_edges'] * hparams['sample_rate'] * 0.001)
        except KeyError:
            delta = 0
        
        all_bout_pd['waveform'] = all_bout_pd.apply(lambda df: wav_i[df['start_sample'] - delta: df['end_sample'] + delta], 
        axis=1)

        all_bout_pd['confusing'] = True
        all_bout_pd['bout_check'] = False
    
    return all_bout_pd, wav_i

def apply_files_offset(sess_pd, hparams):
    # append a column with the absolute timing of the start-end in the day of recording
    # all files assumed to have same length
    logger.debug('Applying file offsets')
    s_f, one_wav = hparams['read_wav_fun'](sess_pd.loc[0]['file'])
    file_len = one_wav.shape[0]
    file_len_ms = file_len // (s_f // 1000)
    
    logger.debug('File len is {}s'.format(file_len_ms*.001))
    sess_pd['i_file'] = sess_pd['file'].apply(hparams['file_order_fun'])
    sess_pd['start_abs'] = sess_pd['start_ms'] + sess_pd['i_file'] * file_len_ms
    sess_pd['end_abs'] = sess_pd['end_ms'] + sess_pd['i_file'] * file_len_ms
    
    sess_pd['start_abs_sample'] = sess_pd['start_sample'] + sess_pd['i_file'] * file_len
    sess_pd['end_abs_sample'] = sess_pd['end_sample'] + sess_pd['i_file'] * file_len
    return sess_pd

def cleanup(bout_pd: pd.DataFrame) -> pd.DataFrame:
    ## check for empty waveforms (how woudld THAT happen???)
    bout_pd['valid_waveform'] = bout_pd['waveform'].apply(lambda x: (False if x.size==0 else True))
    
    # valid is & of all the validated criteria
    bout_pd['valid'] = bout_pd['valid_waveform']
    
    # drop not valid and reset index
    bout_pd.drop(bout_pd[bout_pd['valid']==False].index, inplace=True)
    bout_pd.reset_index(drop=True, inplace=True)
    return bout_pd

def get_bouts_session(raw_folder, proc_folder, hparams, force_p_compute=False):
    logger.info('Going for the bouts in all the files of the session {}'.format(os.path.split(raw_folder)[-1]))
    logger.debug('Saving all process files to {}'.format(proc_folder))
    

    
    sess_files = glob.glob(os.path.join(raw_folder, '*.wav'))
    sess_files.sort()
    logger.debug('Found {} files'.format(len(sess_files)))
    
    all_bout_pd = [pd.DataFrame()]
    for i, raw_file_path in enumerate(sess_files):
        logger.debug('raw file path {}'.format(raw_file_path))
        _, file_name = os.path.split(raw_file_path)
        p_file_path = os.path.join(proc_folder, file_name.split('.wav')[0] + '.npy')
        #logger.debug('p file path {}'.format(p_file_path))
        
        if force_p_compute:
            loaded_p = None
        else:
            try:
                loaded_p = np.load(p_file_path)
            except FileNotFoundError:
                logger.debug('Power file not found, computing')
                loaded_p = None
            except AttributeError:
                logger.debug('No power file path entered, computing')
                loaded_p = None
        
        try:
            bout_pd, _, p = get_bouts_in_file(raw_file_path, hparams, loaded_p=loaded_p)
            bout_pd['file_p'] = p_file_path
            if loaded_p is None:
                logger.debug('Saving p file {}'.format(p_file_path))
                np.save(p_file_path, p)

            all_bout_pd.append(bout_pd)
        except Exception as exc:
            e = sys.exc_info()[0]
            logger.warning('Error while processing {}: {}'.format(raw_file_path, e))
            logger.info(traceback.format_exc)
            logger.info(exc)

    big_pd = pd.concat(all_bout_pd, axis=0, ignore_index=True, sort=True)
    
    ## apply some refinements, filter those that have good waveforms and get spectrograms
    if (hparams['file_order_fun'] is not None) and big_pd.index.size > 0:
        big_pd = apply_files_offset(big_pd, hparams)
        big_pd = cleanup(big_pd)
        logger.info('getting spectrograms')
        big_pd['spectrogram'] = big_pd['waveform'].apply(lambda x: gimmepower(x, hparams)[2])

    out_file = os.path.join(proc_folder, hparams['bout_auto_file'])
    big_pd.to_pickle(out_file)
    fu.chmod(out_file, 0o777)
    logger.info('Saved all to {}'.format(out_file))
    
    return big_pd

def get_epoch_bouts(i_path: str, hparams: dict) -> pd.DataFrame:
    epoch_bout_pd = get_bouts_in_long_file(i_path, hparams)[0]

    i_folder = os.path.split(i_path)[0]
    epoch_bouts_path = os.path.join(i_folder, hparams['bout_auto_file'])
    hparams_pickle_path = os.path.join(i_folder, 'bout_search_params.pickle')

    logger.info('saving bout detect parameters dict to ' + hparams_pickle_path)
    with open(hparams_pickle_path, 'wb') as fh:
        save_param = hparams.copy()
        save_param['read_wav_fun'] = save_param['read_wav_fun'].__name__
        save_param['file_order_fun'] = save_param['file_order_fun'].__name__
        pickle.dump(save_param, fh)
    fu.chmod(hparams_pickle_path, 0o777)

    logger.info('saving bouts pandas to ' + epoch_bouts_path)
    epoch_bout_pd.to_pickle(epoch_bouts_path)
    fu.chmod(epoch_bouts_path, 0o777)

    #epoch_bout_pd = pd.DataFrame()
    return epoch_bout_pd