'''
Generate custom features from chunked preprocessed signal data
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm, glob, time, gc, pickle
from scipy.signal import find_peaks

def atomic_worker(args):

    chunk_path, compute_feats = args

    # setup
    feat_names = []
    feat_arrays = []

    # load preprocessed chunk
    signals = np.load(chunk_path)
    
    # compute desired features

    '''
    Base features
    '''
    if compute_feats['base-feats']:

        peak_ixs, peak_props = find_peaks(signals[0,:], height=2)

    '''
    Aggregate all feats and return as df
    '''

    # Build final pandas dataframe
    df = pd.DataFrame(
        data=np.hstack(feat_arrays),
        columns=feat_names,
    )

    return df


def gen_feats(save_rel_dir, save_name, preprocessed_signals_dir, compute_feats):
    '''
    Generate custom features dataframe from stored preprocessed signal chunks

    :param save_rel_dir (str) Relative dir to save calculated feats
    :param save_name (str) Feat set name
    :param preprocessed_signals_dir (str) Relative dir to preprocessed signals
    :param compute_feats (dict) Dict of bools marking the feats to generate
    :return:
    '''

    np.warnings.filterwarnings('ignore')

    # get sorted paths to all pp signal chunks
    chunk_paths = sorted(glob.glob(preprocessed_signals_dir + '/*'))

    # setup atomic args
    atomic_args = [(cp, compute_feats) for cp in chunk_paths]

    print(f'> feature_engineering : Creating mp pool . . .')

    pool = mp.Pool(processes=mp.cpu_count())
    res = pool.map(atomic_worker, atomic_args)
    pool.close()
    pool.join()

    print(f'> feature_engineering : Concating and saving results . . .')

    # Concat atomic computed feats and save df
    df = pd.concat(res, axis=0)
    df.reset_index(drop=True).to_hdf(save_rel_dir+'/'+save_name, key='w')

    # Also save feature names
    feat_list = list(df.columns)
    with open(save_rel_dir+'/'+save_name.split('.h5')[0]+'.txt', 'w') as f:
        f.writelines([f'{feat_name}\n' for feat_name in feat_list])
    with open(save_rel_dir+'/'+save_name.split('.h5')[0]+'.pkl', 'wb') as f2:
        pickle.dump(feat_list, f2, protocol=pickle.HIGHEST_PROTOCOL)


dataset = 'pp_train_db20'
st = time.time()

compute_feats_template = {
    'base-feats': bool(0),
}

feats_to_gen = {
    'base-feats': 'base-feats_v1_jan14',
}

for ft_name, file_name in feats_to_gen.items():

    cpt_fts = compute_feats_template.copy()
    cpt_fts[ft_name] = True

    gen_feats(
        save_rel_dir='../features',
        save_name=f'{dataset}_{file_name}.h5',
        preprocessed_signals_dir=f'../preprocessed_data/{dataset}',
        compute_feats=cpt_fts,
    )

print(f'> feature_engineering : Done, wall time : {(time.time()-st):.2f} seconds .')

