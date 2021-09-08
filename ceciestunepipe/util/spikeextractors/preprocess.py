import numpy as np
import pandas as pd
import logging

from scipy.io import wavfile
from ceciestunepipe.file import bcistructure as et

from ceciestunepipe.util import sglxutil as sglu
from ceciestunepipe.util import rigutil as ru
from ceciestunepipe.util.spikeextractors.extractors.spikeglxrecordingextractor import spikeglxrecordingextractor as sglex

logger = logging.getLogger('ceciestunepipe.util.spikeextractors.preprocess')

def load_sglx_recordings(exp_struct: dict, epoch:str) -> tuple:
    # get the folders
    # make a pandas of the files
    # read into spikextractors
    # get the rig parameters
    # extract the wav channels of the nidaq signals (mics, syn)

    sgl_folders, sgl_files = sglu.sgl_file_struct(exp_struct['folders']['sglx'])
    #logger.info('loading files {}'.format(sgl_files))
    files_pd = pd.DataFrame(sgl_files)

    i_run = 0 #spikeglx epochs have the one run, this is for later
    run_meta_files = {k: v[i_run] for k, v in sgl_files.items()}
    run_recordings = {k: sglex.SpikeGLXRecordingExtractor(sglu.get_data_meta_path(v)[0]) for k, v in run_meta_files.items()}
    rig_dict = ru.get_rig_par(exp_struct)

    return run_recordings, run_meta_files, files_pd, rig_dict 

def extract_nidq_channels(sess_par, run_recs_dict, rig_dict, chan_name_list, chan_type='adc') -> np.array:
    # get the channels id, numbers for the mic_list
    chan_n_list = [int(ru.lookup_signal(rig_dict, n)[1].split('-')[-1]) for n in chan_name_list]
    
    if chan_type=='adc':
        stream = run_recs_dict['nidq'].get_traces(channel_ids=chan_n_list)
    elif chan_type=='ttl':
        stream = run_recs_dict['nidq'].get_ttl_traces()[chan_n_list, :]
    else:
        raise NotImplementedError('dont know how to deal with {} channels'.format(chan_type))
    
    return stream

def save_wav(stream: np.array, s_f: np.float, wav_path: str) -> int:
    
    wav_s_f = int(round(s_f/1000))*1000
    logger.info('sampling rate {}'.format(wav_s_f))
    
    # write to wav
    logger.info('saving {}-shaped array as wav in {}'.format(stream.shape, wav_path))
    os.makedirs(os.path.split(wav_path)[0], exist_ok=True)
    wavfile.write(wav_path, wav_s_f, stream.T)
    
    ## also save as numpy
    npy_path = wav_path.split('.')[0] + '.npy'
    logger.info('saving {}-shaped array as npy in {}'.format(stream.shape, npy_path))
    np.save(npy_path, stream.T)
    return wav_s_f

def chans_to_wav(recording_extractor, chan_list: list, wav_path:str) -> int:
    # get the stream
    data_stream = recording_extractor.get_traces(channel_ids=chan_list)
    # make sure folder exists
    logger.info('saving {}-shaped array as wav in {}'.format(data_stream.shape, wav_path))
    os.makedirs(os.path.split(wav_path)[0], exist_ok=True)
    # write it
    s_f = int(round(recording_extractor.get_sampling_frequency()/1000))*1000
    logger.info('sampling rate {}'.format(s_f))
    wavfile.write(wav_path, s_f, data_stream.T)
    
    ## also save as numpy
    npy_path = wav_path.split('.')[0] + '.npy'
    logger.info('saving {}-shaped array as npy in {}'.format(data_stream.shape, npy_path))
    np.save(npy_path, data_stream.T)
    return s_f