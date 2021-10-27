
import pandas as pd
import os
import logging
import pickle

import numpy as np

import multiprocessing

import spikeinterface.extractors as se
import spikeinterface.sorters as ss

logger = logging.getLogger('ceciestunepipe.util.spike.kilosort')

N_JOBS_MAX = multiprocessing.cpu_count()-1

def run_spikesort(recording_extractor: se.RecordingExtractor, 
                  logger: logging.Logger,
                  sort_pickle_path: str,
                  tmp_dir: str, 
                  grouping_property: str=None,
                 sorting_method: str='kilosort3',
                 n_jobs_bin: int=N_JOBS_MAX,
                 chunk_mb: int=512, restrict_to_gpu=None,
                 **sort_kwargs):

    logger.info("Grouping property: {}".format(grouping_property))
    logger.info("sorting method: {}".format(sorting_method))
    
    # try:
    if sorting_method == "kilosort2":
        # perform kilosort sorting
        sort_tmp_dir = os.path.join(tmp_dir, 'tmp_ks2')
        logger.info('Sorting tmp dir {}'.format(sort_tmp_dir))
        
        if restrict_to_gpu is not None:
            logger.info('Will set visible gpu devices {}'.format(restrict_to_gpu))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(restrict_to_gpu)

            
        sort = ss.run_kilosort2(
            recording_extractor,
            car=True,
            output_folder=sort_tmp_dir,
            parallel=True,
            verbose=True,
            grouping_property=grouping_property,
            chunk_mb=chunk_mb,
            n_jobs_bin=n_jobs_bin,
            **sort_kwargs
        )
    
    elif sorting_method == "kilosort3":
        # perform kilosort sorting
        sort_tmp_dir = os.path.join(tmp_dir, 'tmp_ks3')
        logger.info('Sorting tmp dir {}'.format(sort_tmp_dir))
        
        if restrict_to_gpu is not None:
            logger.info('Will set visible gpu devices {}'.format(restrict_to_gpu))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(restrict_to_gpu)
            
        sort = ss.run_kilosort3(
            recording_extractor,
            car=True,
            output_folder=sort_tmp_dir,
            parallel=True,
            verbose=True,
            grouping_property=grouping_property,
            chunk_mb=chunk_mb,
            **sort_kwargs
        )
        
    else:
        raise NotImplementedError('Only know how to sort kilosort2/3 for now, \
                                        will deal with {} later'.format(sorting_method))
        
    logger.info('done sorting')
    
    # # save sort
    # logger.info("Saving sort {}".format(sort_pickle_path))
    # with open(sort_pickle_path, "wb") as output:
    #     pickle.dump(sort, output, pickle.HIGHEST_PROTOCOL)
    # logger.info("Sorting output saved to {}".format(sort_pickle_path))
       
    
    # # save sort again with all that processed data
    # sort_temp_pickle_path = sort_pickle_path + '.dump.pkl'
    # logger.info("Saving sort {}".format(sort_temp_pickle_path))
    # sort.dump_to_pickle(sort_temp_pickle_path)
    
    return sort

def load_spikes(ks_folder:str) -> tuple:

    spk_dict = {k: np.load(os.path.join(ks_folder, 
                                        'spike_{}.npy'.format(k))).flatten() for k in ['times', 'clusters']}
    
    spk_dict['cluster_id'] = spk_dict['clusters']
    spk_df = pd.DataFrame(spk_dict)
    clu_df = pd.read_csv(os.path.join(ks_folder, 'cluster_KSLabel.tsv'), 
                                sep='\t', header=0)

    # get the templates
    templ_arr = np.load(os.path.join(ks_folder, 'templates.npy'))
    clu_df['template'] = [x for x in templ_arr]
    
    # with the templates, compute the sorted chanels, main channel, main 7 channels and waveform for the 7 channels
    clu_df['max_chans'] = clu_df['template'].apply(lambda x: np.argsort(np.ptp(x, axis=0))[::-1])
    clu_df['main_chan'] = clu_df['max_chans'].apply(lambda x: x[0])
    
    clu_df['main_7'] = clu_df['max_chans'].apply(lambda x: np.sort(x[:7]))
    clu_df['main_wav_7'] = clu_df.apply(lambda x: x['template'][:, x['max_chans'][:7]], axis=1)
    
    clu_df.sort_values(['KSLabel', 'main_chan'], inplace=True)
    spk_df.sort_values(['times'], inplace=True)

    return clu_df, spk_df

def load_cluster_info(ks_folder: str) -> pd.DataFrame:
    info_df = pd.read_csv(os.path.join(ks_folder, 'cluster_info.tsv'), 
                            sep='\t', header=0)

    return info_df

def get_window_spikes(spk_df, clu_list, start_sample, end_sample):
    onset = start_sample
    offset = end_sample
    
    spk_t = spk_df.loc[spk_df['times'].between(onset, offset, inclusive=False)]
    
    spk_arr = np.zeros((clu_list.size, offset - onset))

    for i, clu_id in enumerate(clu_list):
        clu_spk_t = spk_t.loc[spk_t['clusters']==clu_id, 'times'].values
        spk_arr[i, clu_spk_t - onset] = 1
    return spk_arr
    
def get_rasters(spk_df, clu_list, start_samp_arr, span_samples):
    # returns np.array([n_clu, n_sample, n_trial])
    
    # get the window spikes for all of the clusters, for each of the start_samp_arr
    spk_arr_list = [get_window_spikes(spk_df, clu_list, x, x+span_samples) for x in start_samp_arr]
    return np.stack(spk_arr_list, axis=-1)


