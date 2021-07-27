
import pandas as pd
import os
import logging
import numpy as np

logger = logging.getLogger('ceciestunepipe.util.spike.kilosort')

def load_spikes(ks_folder:str) -> tuple:


    spk_dict = {k: np.load(os.path.join(ks_folder, 
                                        'spike_{}.npy'.format(k))).flatten() for k in ['times', 'clusters']}
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