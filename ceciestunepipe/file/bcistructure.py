from builtins import NotImplementedError
import os
import shutil
import socket
import json
import logging
import glob
import warnings
import pandas as pd

from ceciestunepipe.util import fileutil as fu

logger = logging.getLogger('ceciestunepipe.file.bcistructure')

locations_dict = dict()

# locations_dict['passaro'] = {'mnt': os.path.abspath('/mnt/cube/earneodo/basalganglia/birds'),
#                              'local': os.path.abspath('/Data/basalganglia'),
#                              'fast': os.path.abspath('/Data/scratch')
#                              }

locations_dict['passaro']= {'mnt': os.path.abspath('/mnt/cube/earneodo/basalganglia/birds'),
                             'local': os.path.abspath('/Data/raw_data'),
                             'fast': os.path.abspath('/mnt/cube/earneodo/scratch')
                             }

locations_dict['txori']= {'mnt': os.path.abspath('/mnt/sphere/speech_bci/'),
                             'local': os.path.abspath('/scratch/earneodo'),
                             'fast': os.path.abspath('/scratch/earneodo')
                             }

locations_dict['pakhi']= {'mnt': os.path.abspath('/mnt/sphere/speech_bci/'),
                             'local': os.path.abspath('/scratch/earneodo'),
                             'fast': os.path.abspath('/scratch/earneodo')
                             }

locations_dict['pouli']= {'mnt': os.path.abspath('/mnt/sphere/speech_bci/'),
                             'local': os.path.abspath('/experiment/'),
                             'fast': os.path.abspath('/experiment/tmp')
                             }

# locations_dict['pakhi']= {'mnt': os.path.abspath('/mnt/cube/earneodo/bci_zf/neuropix/birds'),
#                              'local': os.path.abspath('/mnt/sphere/earneodo/bci_zf/ss_data'),
#                              'fast': os.path.abspath('/scratch/earneodo/tmp')
#                              }


# locations_dict['lookfar'] = {'mnt': os.path.abspath('/Users/zeke/experiment/birds'),
#                              'local': os.path.abspath('/Users/zeke/experiment/birds'),
#                              'fast': os.path.abspath('/Users/zeke/experiment/scratch')
#                              }

## Added this to run analyses locally
locations_dict['jupyter-lostrowski']= {'mnt': os.path.abspath('/net2/expData/birdsong_var/zebra_finch'),
                             'local': os.path.abspath('/net2/expData/birdsong_var/zebra_finch'),
                             'fast': os.path.abspath('/net2/expData/birdsong_var/zebra_finch')
                             }

default_struct_par = {'neural': 'ephys',
                      'presort': 'kwik',  
                      'sort': 'msort'}
# Todo: define locations_dict in a json file or something more elegant


def get_locations_from_hostname():
    hostname = socket.gethostname().split('.')[0]
    return locations_dict[hostname]


def read_json_exp_struct():
    raise NotImplementedError

def get_location_dict() -> dict:
    # get the locations dict from an environment variable or from the hostname, using the get_locations_from_hostname
    read_exp_base = os.environ.get('EXPERIMENT_PATH')

    if read_exp_base is not None:
        # if there is a .json file configured with the variables of the experiment
        exp_base = os.path.abspath(read_exp_base)
        location_dict_json_path = os.path.join(exp_base, 'exp_struct.json')
        location_dict = read_json_exp_struct()
    else:
        # try to read it from the hostname
        location_dict = get_locations_from_hostname()
    return location_dict

def get_file_structure(location: dict, sess_par: dict) -> dict:
    """[summary]
    Arguments:
        location {dict} -- [description]
        sess_par {dict} -- session parameters dictionary. Example:
            sess_par = {'bird': 'p1j1',
            'sess': '2019-02-27_1800_02', 
            'probe': 'probe_0',
            'sort': 0} 
            - bird and sess are self-explanatory and refer to the folder of the raw, kwd files.
            - probe describes the probe that was used to do the sorting, which in turn determines
              neural port (via the rig.json file) and probe mapping (via rig.json file and 
              pipefinch.pipeline.filestructure.probes)
            - sort determines the version of sorting, in case multiple sorts were made on the same
              session (e.g different time spans, different configurations of the sorting algorithm)
              if the field is not present or is None, the .kwik, unit_waveforms and rasters will be directly
              in the Ephys\kwik\sess_id folder.
              otherwise, a 'sort_x' folder will contain the sorting files.

    Returns:
        dict -- ditcionary containing paths of folders and files.
            exp_struct['folders']: dictionary with ['raw', 'kwik', 'msort'] keys
            exp_struct['files']: dictionary with keys:
                'par': expermient.json path
                'set': settings.isf intan generated file path
                'rig': rig.json file desribing the recording settings (channels/signals)

                'kwd': kwd file with all raw data from the session
                'kwik': kwik file with sorted spikes
                'kwe': 
    """

    try:
        ephys_folder = sess_par['ephys_software']
    except KeyError:
        logger.info('ephys folder defaults to sglx')
        ephys_folder = 'sglx'
    
    exp_struct = {}
    bird, sess = sess_par['bird'], sess_par['sess']

    exp_struct['folders'] = {}
    exp_struct['files'] = {}

    # The bird structure
    exp_struct['folders']['bird'] = os.path.join(location['mnt'], 'raw_data', bird)
    
    # The raw files
    exp_struct['folders']['raw'] = os.path.join(
        location['mnt'], 'raw_data', bird, sess)
    
    exp_struct['folders'][ephys_folder] = os.path.join(exp_struct['folders']['raw'], ephys_folder)
    for f, n in zip(['par', 'set', 'rig'],
                    ['experiment.json', 'settings.isf', 'rig.json']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders'][ephys_folder], n)

    # the kwik system (spikes, events, kwd file with streams)
    exp_struct['folders']['kwik'] = os.path.join(
        location['local'], bird, ephys_folder, 'kwik', sess)
    for f, n in zip(['kwd', 'kwik', 'kwe'], ['stream.kwd', 'spikes.kwik', 'events.kwe']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['kwik'], n)

    if 'sort' in sess_par and sess_par['sort'] is not None:
        exp_struct['files']['kwik'] = os.path.join(exp_struct['folders']['kwik'],
                                                   'sort_{}'.format(sess_par['sort']), 'spikes.kwik')

    # the processed system (dat_mic.mat, dat_ap.mat et. al files)
    exp_struct['folders']['processed'] = os.path.join(
        location['mnt'], 'processed_data', bird, sess, ephys_folder)
    for f, n in zip(['dat_mic', 'dat_ap', 'allevents'],
                    ['dat_mic.mat', 'dat_ap.mat', 'dat_all.pkl']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['processed'], n)

    # the 'derived' system (wav_mic, ...)
    exp_struct['folders']['derived'] = os.path.join(
        location['mnt'], 'derived_data', bird, sess, ephys_folder)
    for f, n in zip(['wav_mic'], ['wav_mic.wav']):
        exp_struct['files'][f] = os.path.join(exp_struct['folders']['derived'], n)

    # the aux, temporary mountainsort files. these will be deleted after sorting
    # try 'fast' location first, if it does not exist, go for 'local'
    try:
        exp_struct['folders']['tmp'] = os.path.join(location['fast'], 'tmp')
    except KeyError:
        exp_struct['folders']['tmp'] = os.path.join(location['local'], 'tmp')
    
     # SET THE TMP DIRECTORY ENVIRONMENT VARIABLEM
    os.environ["TMPDIR"] = exp_struct['folders']['tmp']
    os.environ["TMP"] = exp_struct['folders']['tmp']

    try:
        msort_location = location['fast']
    except KeyError:
        msort_location = location['local']



    # MOUNTAINSORT FILE STRUCTURE
    exp_struct['folders']['msort'] = os.path.join(
        msort_location, bird, ephys_folder, 'msort', sess)
    for f, n in zip(['mda_raw', 'par'], ['raw.mda', 'params.json']):
        exp_struct['files'][f] = os.path.join(
            exp_struct['folders']['msort'], n)

    # KILOSORT FILE STRUCTURE
    # determine sort version, both for ksort and for the sort result in derived data
    try:
        sort_version = str(sess_par['sort'])
        if sort_version is None:
            sort_version = ''
    except KeyError:
        sort_version = ''
        
    exp_struct['folders']['ksort'] = os.path.join(
        msort_location, bird, ephys_folder, 'ksort', sess, sort_version)

    for f, n in zip(['bin_raw', 'par'], ['raw.bin', 'params.json']):
        exp_struct['files'][f] = os.path.join(
            exp_struct['folders']['ksort'], n)

    # Sorted data file structure
    exp_struct['folders']['sort'] = os.path.join(exp_struct['folders']['derived'], sort_version)
    return exp_struct


def get_exp_struct(bird, sess, ephys_software='sglx', sort='', location_dict: dict = dict()):
    # get the configuration of the experiment:
    # if environment variable 'EXPERIMENT_PATH' exists,
    # read 'EXPERIMENT_PATH/config/expstruct.json'
    # no location dict was entered, try to get it from the hostname (from locations_dict)
    if location_dict:
        pass
    else:
        location_dict = get_location_dict()

    # make the exp struct dict.
    sess_par_dict = {'bird': bird,
                     'sess': sess,
                     'sort': sort,
                     'ephys_software': ephys_software}
    exp_struct = get_file_structure(location_dict, sess_par_dict)

    return exp_struct

def get_birds_list(ephys_software, location_dict: dict=dict()):
    raise NotImplementedError

def get_rig_par(exp_struct: dict) -> dict:
    rig_par_file = exp_struct['files']['rig']
    with open(rig_par_file, 'r') as fp:
        rig_par = json.load(fp)
    return rig_par

def make_all_events(exp_struct: dict) -> dict:
    mic_dat_path = exp_struct['files']['dat_mic']
    

def get_probe_port(exp_struct: dict, selected_probe: str) -> str:
    # get the probe and the port where the probe was connected
    rig_par = get_rig_par(exp_struct)
    probe_port = rig_par['chan']['port'][selected_probe].strip('-')
    return probe_port

def list_sessions(bird: str, location_dict: dict = dict(), section='raw', ephys_software='sgl') -> list:
    exp_struct = get_exp_struct(bird, '', location_dict=location_dict)
    try:
        sess_list = next(os.walk(exp_struct['folders'][section]))[1]
        sess_list.sort()
        return sess_list
    except StopIteration:
        #raise Warning
        msg = 'No sessions for bird in {}'.format(exp_struct['folders'][section])
        warnings.warn(msg)
        #return None

def list_birds(location_dict: dict = dict(), section='raw', ephys_software='sgl') -> list:
    exp_struct = get_exp_struct('', '', location_dict=location_dict)
    try:
        sess_list = next(os.walk(exp_struct['folders'][section]))[1]
        sess_list.sort()
        return sess_list
    except StopIteration:
        #raise Warning
        msg = 'No sessions for bird in {}'.format(exp_struct['folders'][section])
        warnings.warn(msg)
        #return None

def msort_cleanup(exp_struct: dict):
    # remove the mda files
    mda_raw_path = exp_struct['files']['mda_raw']
    logger.info('removing (if exists) msort mda file {} '.format(mda_raw_path))
    try:
        os.remove(mda_raw_path)
    except FileNotFoundError:
        logger.debug('Nuttin done, file wasnt there')

def save_sort(tmp_loc, sort_folder, exist_ok=False, exclude_list=['*.dat']):
    ### save all the temp files (except for the heavy ones) to the final dest
    fu.makedirs(sort_folder)
    #file_list, exclude_list = fu.glob_except(tmp_loc, exclude_list=exclude_list)
    logger.info('Copying the temp files from sort in {} to {}'.format(tmp_loc, sort_folder)) 
    return shutil.copytree(tmp_loc, os.path.split(sort_folder)[0], 
                    ignore=shutil.ignore_patterns(*exclude_list),
                    dirs_exist_ok=exist_ok)

def list_subfolders(folder_path):
    return next(os.walk(os.path.abspath(folder_path)))[1]


def sgl_struct(sess_par: dict, epoch: str, ephys_software='sglx') -> dict:
    # locations of the folders for the epoch, if ephys is sglx
    exp_struct = get_exp_struct(
        sess_par['bird'], sess_par['sess'], ephys_software=ephys_software, sort='')
    
    

    ### for most, the epoch goes last
    exp_struct['folders'] = {k: os.path.join(v, epoch)
                  for k, v in exp_struct['folders'].items()}
    
    ### for the sort, the sort goes last
    epoch_first_keys = ['sort', 'ksort']
    sort_version = str(sess_par['sort'])
    epoch_first_keys_folders_dict = {k: os.path.join(v, sort_version)
                  for k, v in exp_struct['folders'].items() if (k in epoch_first_keys)}
    exp_struct['folders'].update(epoch_first_keys_folders_dict)


    update_files = ['kwd', 'kwe', 'mda_raw', 'bin_raw', 'kwik', 'par', 'wav_mic']
    updated_files_dict = {k: os.path.join(os.path.split(v)[0],
                                          epoch,
                                          os.path.split(v)[-1]) for k, v in exp_struct['files'].items() if k in update_files}
    
    exp_struct['files'].update(updated_files_dict)
    exp_struct['files']['kwik'] = os.path.join(os.path.split(exp_struct['files']['kwik'])[0],
                                               'sort_{}'.format(sess_par['sort']),
                                               os.path.split(exp_struct['files']['kwik'])[-1])
    
    #logger.info(updated_files_dict)
    
    return exp_struct

def list_sgl_epochs(sess_par: dict, raw_paths=False, location_dict: dict = dict()) -> list:
    # points to all the epochs in a session
    exp_struct = get_exp_struct(
        sess_par['bird'], sess_par['sess'], 'sglx', sess_par['sort'],
        location_dict=location_dict)

    kwik_folder = exp_struct['folders']['kwik']
    raw_folder = exp_struct['folders']['sglx']

    logger.info(exp_struct)

    all_sess_folders = list(
        filter(os.path.isdir, glob.glob(os.path.join(raw_folder, '*'))))
    all_sess_folders.sort()
    if raw_paths:
        return all_sess_folders
    else:
        return list(map(lambda x: os.path.split(x)[-1], all_sess_folders))

def list_ephys_epochs(sess_par: dict, raw_paths=False, location_dict: dict = dict()) -> list:
    # points to all the epochs in a session
    exp_struct = get_exp_struct(
        sess_par['bird'], sess_par['sess'], sess_par['ephys_software'], sess_par['sort'],
        location_dict=location_dict)

    kwik_folder = exp_struct['folders']['kwik']
    raw_folder = exp_struct['folders'][sess_par['ephys_software']]

    logger.info(exp_struct)

    all_sess_folders = list(
        filter(os.path.isdir, glob.glob(os.path.join(raw_folder, '*'))))
    all_sess_folders.sort()
    if raw_paths:
        return all_sess_folders
    else:
        return list(map(lambda x: os.path.split(x)[-1], all_sess_folders))

def get_sgl_files_epochs(parent_folder, file_filter='*.wav'):
    sess_files = []
    for root, subdirs, files in os.walk(parent_folder):
        for epoch_dir in subdirs:
            #print(glob.glob(os.path.join(root, epoch_dir, file_filter)))
            sess_files += list(glob.glob(os.path.join(root, epoch_dir, file_filter)))
    sess_files.sort()
    return sess_files


def split_path(path:str) -> list:
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def epoch_meta_from_bout_series(s: pd.Series):
    bird, sess, _, epoch = split_path(s['file'])[-5:-1]
    s['bird'] =  bird
    s['sess'] = sess
    s['epoch'] = epoch
    return s

def get_epoch_bout_pd(sess_par, only_curated=False, software='sglx'):
    epoch = sess_par['epoch']
    if software == 'sglx':
        exp_struct = sgl_struct(sess_par, epoch)
        sess_derived_path = os.path.split(os.path.split(exp_struct['folders']['derived'])[0])[0]
        bout_pd_path = os.path.join(sess_derived_path, 'bouts_sglx', 'bout_curated.pickle')
    
    elif software == 'oe':
        exp_struct = get_exp_struct(sess_par['bird'], sess_par['sess'], sort=sess_par['sort'], ephys_software='bouts_oe')
        bout_pd_path = os.path.join(exp_struct['folders']['derived'], 'bout_curated.pickle')

    logger.info('loading curated bouts for session {} from {}'.format(sess_par['sess'],
                                                                               bout_pd_path))
    
    bout_pd = pd.read_pickle(bout_pd_path)
    
    # fill in with epoch metadata from the wav files path
    logger.info('Filtering bouts for epoch {}'.format(epoch))
    bout_pd = bout_pd.apply(epoch_meta_from_bout_series, axis=1)
    
    # drop all rows from other epoch
    if only_curated:
        logger.info('Filtering also only manually curated bouts')
        bout_pd = bout_pd[(bout_pd['epoch'] == epoch) & (bout_pd['bout_check'] == True)]
        bout_pd.reset_index(drop=True, inplace=True)
    else:
        bout_pd = bout_pd[(bout_pd['epoch'] == epoch)]
        bout_pd.reset_index(drop=True, inplace=True)
        
    return bout_pd
