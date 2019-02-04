'''
Generate custom features from chunked preprocessed signal data
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm, glob, time, pickle, re
from itertools import product
from false_pos_suppression import fps

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

        # Feature names
        base_feats_names = [
            'num_peaks',
            'mean_height',
            'std_height',
            'std_width',
            'percen_width_10',
            'percen_width_90',
            'assymetry',
        ]

        # FPS params
        ratio_ranges = [0.25]
        max_distances = [30]
        rel_heights = [0.1]

        for ratio_range, max_distance, rel_height in product(ratio_ranges, max_distances, rel_heights):

            suffix = f'_rr{ratio_range:.2f}_md{max_distance:d}_rl{rel_height:.2f}'
            feat_names.extend([f'{name}{suffix}' for name in base_feats_names])
            num_base_feats = len(base_feats_names)

            # Feature array
            base_feats_array = np.zeros(shape=(signals.shape[0], num_base_feats))

            for i, signal in tqdm.tqdm(enumerate(signals), total=signals.shape[0]):

                # Extract peak properties
                peak_heights, peak_widths, peak_ixs = fps(
                    signal=signal,
                    min_height=2,
                    max_height=20,
                    ratio_range=ratio_range,
                    max_distance=max_distance,
                    clean_distance=500,
                    rel_height=rel_height,
                )

                # Compute feats
                feat_list = [
                    peak_heights.size,

                    np.mean(peak_heights) if len(peak_heights) != 0 else np.nan,
                    np.std(peak_heights) if len(peak_heights) != 0 else np.nan,

                    np.std(peak_widths) if len(peak_widths) != 0 else np.nan,

                    np.percentile(peak_widths, 10) if len(peak_widths) != 0 else np.nan,
                    np.percentile(peak_widths, 90) if len(peak_widths) != 0 else np.nan,

                    np.abs(np.mean(peak_heights[peak_heights > 0]) / np.mean(peak_heights[peak_heights > 0])) if len(
                        peak_heights) != 0 else np.nan,
                ]
                for k, feat in enumerate(feat_list):
                    base_feats_array[i, k] = feat

            feat_arrays.append(base_feats_array)

    '''
    Nofps features
    '''
    if compute_feats['nofps-feats']:

        # Feature names
        base_feats_names = [
            'a'
        ]

        # FPS params
        ratio_ranges = [0.25]
        max_distances = [30]
        rel_heights = [0.1]

        for ratio_range, max_distance, rel_height in product(ratio_ranges, max_distances, rel_heights):

            suffix = f'_rr{ratio_range:.2f}_md{max_distance:d}_rl{rel_height:.2f}'
            feat_names.extend([f'{name}{suffix}' for name in base_feats_names])
            num_base_feats = len(base_feats_names)

            # Feature array
            base_feats_array = np.zeros(shape=(signals.shape[0], num_base_feats))

            for i, signal in tqdm.tqdm(enumerate(signals), total=signals.shape[0]):

                # Extract peak properties
                peak_heights, peak_widths, peak_ixs = fps(
                    signal=signal,
                    min_height=2,
                    max_height=20,
                    ratio_range=ratio_range,
                    max_distance=max_distance,
                    clean_distance=500,
                    rel_height=rel_height,
                )

                # Compute feats
                feat_list = [
                    peak_heights.size,

                    np.mean(peak_heights) if len(peak_heights) != 0 else np.nan,
                    np.std(peak_heights) if len(peak_heights) != 0 else np.nan,

                    np.std(peak_widths) if len(peak_widths) != 0 else np.nan,

                    np.percentile(peak_widths, 10) if len(peak_widths) != 0 else np.nan,
                    np.percentile(peak_widths, 90) if len(peak_widths) != 0 else np.nan,
                ]
                for k, feat in enumerate(feat_list):
                    base_feats_array[i, k] = feat

            feat_arrays.append(base_feats_array)

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
    chunk_paths = glob.glob(preprocessed_signals_dir + '/*') # unsorted paths
    chunk_suffixes = [re.search('_\d+\.npy', chunk_name).group() for chunk_name in chunk_paths] # get suffixes
    chunk_numbers = np.array([int(suffix[1:-4]) for suffix in chunk_suffixes]) # grab number only from each suffix
    arg_sort = np.argsort(chunk_numbers) # get correct order
    chunk_paths = [chunk_paths[i] for i in arg_sort] # apply to path list to put it in correct order

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
    'nofps-feats': bool(0),
}

feats_to_gen = {
    'base-feats': 'base-feats_v22',
    # 'nofps-feats': 'nofps-feats_v21',
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

