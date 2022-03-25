from xmlrpc.client import boolean
from pickle import Unpickler
from builtins import FileExistsError, FileNotFoundError, NotImplementedError
import more_itertools as mit
import warnings
import traceback
import numpy as np
import pandas as pd
import pickle
import logging
import os
import glob
import socket
import sys
import datetime

from joblib import Parallel, delayed
from scipy.io import wavfile
from tqdm.auto import tqdm

from ceciestunepipe.util.sound import spectral as sp
from ceciestunepipe.util.sound import temporal as st

from ceciestunepipe.util import fileutil as fu

from ceciestunepipe.file import bcistructure as et
from ceciestunepipe.util.sound import boutsearch as bs

logger = logging.getLogger('ceciestunepipe.pipeline.searchbout')


class BoutParamsUnpickler(pickle.Unpickler):
    # hparams during the search in boutsearch contains functions that are saved in the pickle.
    # Loading the pickle naively will search for those functions in the __main__ module and will fail.
    # This custom pickle loader will interject and replace the functions with the ones in the boutsearch module.
    # https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
    def find_class(self, module, name):
        if name == 'read_wav_chan':
            from ceciestunepipe.util.sound import boutsearch as bs
            return bs.read_wav_chan
        elif name == 'sess_file_id':
            from ceciestunepipe.util.sound import boutsearch as bs
            return bs.sess_file_id
        else:
            return super().find_class(module, name)


def get_all_day_bouts(sess_par: dict, hparams: dict, ephys_software='alsa', n_jobs: int = 12) -> pd.DataFrame:

    logger.info('Will search for bouts through all session {}, {}'.format(
        sess_par['bird'], sess_par['sess']))
    exp_struct = et.get_exp_struct(
        sess_par['bird'], sess_par['sess'], ephys_software=ephys_software)

    # get all the paths to the wav files of the day
    raw_folder = exp_struct['folders'][ephys_software]
    wav_path_list = glob.glob(os.path.join(
        exp_struct['folders'][ephys_software], '*.wav'))
    wav_path_list.sort()
    logger.info('Found {} files'.format(len(wav_path_list)))

    def get_file_bouts(i_path):
        return bs.get_bouts_in_file(i_path, hparams)[0]

    # Go parallel through all the paths in the day, get a list of all the pandas dataframes for each file
    sess_pd_list = Parallel(n_jobs=n_jobs, verbose=100, backend=None)(
        delayed(get_file_bouts)(i) for i in wav_path_list)

    #sess_pd_list = [bs.get_bouts_in_file(i, hparams)[0] for i in wav_path_list]
    # update the sample rate
    hparams['sample_rate'] = bs.sample_rate_from_wav(wav_path_list[0])
    # concatenate the file, filter by 'good' waveform, get spectrogram and reindex
    sess_bout_pd = pd.concat(sess_pd_list)
    # before doing check that it's not empty
    if sess_bout_pd.index.size > 0:
        sess_bout_pd = bs.cleanup(sess_bout_pd)
        if sess_bout_pd.index.size > 0:
            logger.info('getting spectrograms')
            sess_bout_pd['spectrogram'] = sess_bout_pd['waveform'].apply(lambda x: bs.gimmepower(x, hparams)[2])
        else:
            logger.info('Bouts dataframe came out empty after cleaning up')
    else:
        logger.info('Bouts dataframe came out empty after search')
    
    #finally, save
    sess_bout_pd.reset_index(drop=True, inplace=True)
    save_auto_bouts(sess_bout_pd, sess_par, hparams)
    return sess_bout_pd


def save_auto_bouts(sess_bout_pd, sess_par, hparams):
    exp_struct = et.get_exp_struct(
        sess_par['bird'], sess_par['sess'], ephys_software='alsa')
    sess_bouts_dir = os.path.join(
        exp_struct['folders']['derived'], 'bouts_ceciestunepipe')

    sess_bouts_path = os.path.join(sess_bouts_dir, hparams['bout_auto_file'])
    hparams_pickle_path = os.path.join(
        sess_bouts_dir, 'bout_search_params.pickle')

    fu.makedirs(sess_bouts_dir, exist_ok=True, mode=0o777)
    logger.info('saving bouts pandas to ' + sess_bouts_path)
    sess_bout_pd.to_pickle(sess_bouts_path)
    fu.chmod(sess_bouts_path, 0o777)

    logger.info('saving bout detect parameters dict to ' + hparams_pickle_path)
    with open(hparams_pickle_path, 'wb') as fh:
        pickle.dump(hparams, fh)
    fu.chmod(hparams_pickle_path, 0o777)


def bout_to_wav(a_bout: pd.Series, sess_par, hparams, dest_dir):
    file_name = '{}_{}_{}.wav'.format(sess_par['sess'],
                                      os.path.split(
                                          a_bout['file'])[-1].split('.wav')[0],
                                      a_bout['start_ms'])
    file_path = os.path.join(dest_dir, file_name)
    x = a_bout['waveform']

    wavfile.write(file_path, hparams['sample_rate'], x)
    return file_path


def bouts_to_wavs(sess_bout_pd, sess_par, hparams, dest_dir):
    # make the dest_dir if does not exist
    logger.info('Saving all session bouts to folder ' + dest_dir)
    fu.makedirs(dest_dir, exist_ok=True, mode=0o777)
    # write all the motifs to wavs
    sess_bout_pd.apply(lambda x: bout_to_wav(
        x, sess_par, hparams, dest_dir), axis=1)
    # write the hparams as pickle
    hparams_pickle_path = os.path.join(
        dest_dir, 'bout_search_params_{}.pickle'.format(sess_par['sess']))

    logger.info('saving bout detect parameters dict to ' + hparams_pickle_path)
    with open(hparams_pickle_path, 'wb') as fh:
        pickle.dump(hparams, fh)


def has_bouts_file(bird: str, sess: str, ephys_software: str, derived_folder: str = 'bouts_ceciestunepipe',
                bout_type: str = 'bout_auto_file') -> bool:
    
    exp_struct = et.get_exp_struct(bird, sess, ephys_software=ephys_software)
    bouts_folder = os.path.join(
        exp_struct['folders']['derived'], derived_folder)
    hparams_file_path = os.path.join(bouts_folder, 'bout_search_params.pickle')

    try:
        with open(hparams_file_path, 'rb') as fh:
            unpickler = BoutParamsUnpickler(fh)
            hparams = unpickler.load()

        bouts_auto_file_path = os.path.join(bouts_folder, hparams[bout_type])
        has_bout_file = True if os.path.isfile(bouts_auto_file_path) else False

    except FileNotFoundError:
        logger.info('Search/bouts file not found in {}'.format(bouts_folder))
        has_bout_file = False
    return has_bout_file


def load_bouts(bird: str, sess: str, ephys_software: str, derived_folder: str = 'bouts_ceciestunepipe', bout_file_key='bout_auto_file') -> pd.DataFrame:

    exp_struct = et.get_exp_struct(bird, sess, ephys_software=ephys_software)
    bouts_folder = os.path.join(
        exp_struct['folders']['derived'], derived_folder)

    # try loading the params, bouts file
    # if exist, return that one, otherwise return None
    hparams_file_path = os.path.join(bouts_folder, 'bout_search_params.pickle')

    try:
        #read_wav_chan = copy_func(bs.read_wav_chan)
        with open(hparams_file_path, 'rb') as fh:
            unpickler = BoutParamsUnpickler(fh)
            hparams = unpickler.load()
        bouts_auto_file_path = os.path.join(
            bouts_folder, hparams[bout_file_key])
        # load. It is important to reset index because the manual curation requires unique indexing
        bouts_pd = pd.read_pickle(bouts_auto_file_path).reset_index(drop=True)
    except FileNotFoundError:
        logger.info('Search/bouts file not found in {}'.format(bouts_folder))
        bouts_pd = None
        hparams = None
    return hparams, bouts_pd


def search_bird_bouts(bird: str, sess_list: list, hparams:dict, ephys_software: str='alsa', n_jobs=4, force=False) -> int:
    logger.info('Getting all bouts in bird {} sessions {}'.format(bird, sess_list))

    # check if bouts exist for the sessions for this bird 
    for sess in sess_list:
        sess_par = {'bird': bird, 'sess': sess}
        if (not force) and has_bouts_file(bird, sess, ephys_software=ephys_software):
            logger.info('Bird {} already had a bouts file in sess {}'.format(bird, sess))
        else:
            logger.info('Will search bouts for Bird {} - sess {}'.format(bird, sess))
            try:
                sess_bout_pd = get_all_day_bouts(sess_par, hparams, ephys_software= ephys_software, n_jobs=n_jobs)
            except:
                logger.info('Error on bird {} - sess {}'.format(bird, sess))
                traceback.print_exc()
    
    return 0

def all_bird_bouts_search(bird: str, days_lookup: int, hparams:dict, ephys_software: str='alsa', n_jobs=4, force=False, do_today=False) -> int:
    # get all bouts in a bird for the last days_lookup days
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    from_date = datetime.date.today() - datetime.timedelta(days=days_lookup)
    from_date_str = from_date.strftime('%Y-%m-%d')
    
    logger.info('Getting all bouts for bird {} from date {} onward'.format(bird, from_date_str))
    if do_today:
        logger.info('Including today')

    sess_list = et.list_sessions(bird, section='raw', ephys_software=ephys_software)
    sess_arr = np.array(sess_list)
    
    sess_arr = sess_arr[(sess_arr >= from_date_str)]

    if do_today is False:
        sess_arr = sess_arr[:-1]

    do_sess_list = list(sess_arr)
    search_result = search_bird_bouts(bird, do_sess_list, hparams, ephys_software=ephys_software, n_jobs=n_jobs, force=force)
    return search_result

def get_birds_bouts(birds_list: list, days_lookup: int, hparams:dict, ephys_software: str='alsa', n_jobs=4, force=False, do_today=False) -> int:
    logger.info('Getting all bouts for birds {} for the last {} days'.format(birds_list, days_lookup))

    for bird in birds_list:
        search_result = all_bird_bouts_search(bird, days_lookup, hparams, ephys_software=ephys_software, n_jobs=n_jobs, force=force, do_today=do_today)
    return search_result

def get_starlings_alsa_bouts(days_lookup: int, n_jobs: int=4, force: boolean=False, do_today=False) -> int:
    # get all birds
    all_birds_list = et.list_birds(section='raw', ephys_software='alsa')

    # apply a filter to get the starlings
    starling_list = [b for b in all_birds_list if b.split('_')[0] == 's']

    # get all_sessions for the bird list
    starling_hparams = bs.default_hparams
    get_birds_bouts(starling_list, days_lookup, starling_hparams, ephys_software='alsa', n_jobs=n_jobs, force=force, do_today=do_today)

def get_one_day_bouts(bird, sess):
    starling_hparams = bs.default_hparams
    search_result = search_bird_bouts(bird, [sess], starling_hparams, ephys_software='alsa', n_jobs=12)

def read_session_bouts(bird: str, sess: str, recording_software='alsa', curated: boolean=False) -> pd.DataFrame:
    # get the location of the sess, parameters pd.
    # if has spectrograms do nothing, otherwise compute
    # reindex/reset index
    # reset the manual labels
    #exp_struct = et.get_exp_struct(bird, sess, ephys_software=recording_software)
    #derived_folder = exp_struct['folders']['derived']
    #logger.info('Looking for params, bout pickle file in folder {}'.format(derived_folder))
    
    bout_file_key = 'bout_curated_file' if curated else 'bout_auto_file'
    hparams, bout_pd = load_bouts(bird, sess, recording_software, derived_folder = 'bouts_ceciestunepipe', bout_file_key=bout_file_key)
    if not curated:
        # this could be something (none or empty) or none.
        bout_pd = pd.DataFrame() if bout_pd is None else bout_pd
        if bout_pd.index.size > 0:
            bs.cleanup(bout_pd)
        #if it comes up nonempty after cleanup, get spectrograms, etc
        if bout_pd.index.size >0:
        # if it is something:
            if hparams['sample_rate'] is None:
                one_wav_path = bout_pd.loc[0, 'file']
                logger.info('Sample rate not saved in parameters dict, searching it in ' + one_wav_path)
                hparams['sample_rate'] = bs.sample_rate_from_wav(one_wav_path)
            if not ('spectrogram' in bout_pd.keys()):
                logger.info('No spectrograms in here, will compute...')
                bout_pd['spectrogram'] = bout_pd['waveform'].apply(lambda x: bs.gimmepower(x, hparams)[2])
                #logger.info('saving bout pandas with spectrogram to ' + bouts_auto_file_path)

            if not ('confusing' in bout_pd.keys()):
                bout_pd['confusing'] = True

            if not ('bout_check' in bout_pd.keys()):
                bout_pd['bout_check'] = False
            bout_pd.reset_index(drop=True, inplace=True)
            # check whether it has the spectrogram and compute it if it doesnt compute them
            # reset the index
            # add the curation tags
    return hparams, bout_pd


def main():
    """Launcher"""
    # make a logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info('Running searchbout on {}'.format(socket.gethostname()))

    # get all birds
    days_lookup = 2
    do_today=False
    force_compute = False

    get_starlings_alsa_bouts(days_lookup, force=force_compute, do_today=do_today)

    #get_birds_bouts(['s_b1370_22'], days_lookup, bs.default_hparams, ephys_software='alsa', n_jobs=12, force=True, do_today=do_today)
    #get_one_day_bouts('s_b1267_22', '2022-03-24')
    # apply filters if any

    logger.info('done for the day')
    return 0


if __name__ == "__main__":
    sys.exit(main())
