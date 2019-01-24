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
            'mean_width',
            'std_width',
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
                peak_heights, peak_widths = fps(
                    signal=signal,
                    min_height=4,
                    max_height=40,
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

                    np.mean(peak_widths) if len(peak_widths) != 0 else np.nan,
                    np.std(peak_widths) if len(peak_widths) != 0 else np.nan,
                ]
                for k, feat in enumerate(feat_list):
                    base_feats_array[i, k] = feat

            feat_arrays.append(base_feats_array)

    '''
    Quarter features
    '''
    if compute_feats['quarter-feats']:

        # Split each signal in 'num_slices'
        num_slices = 4
        slice_size = int(signals.shape[1] / num_slices)

        # Feature names
        quarter_feats_names = [
            'cross_num_peaks',
            'cross_mean_height',
            'cross_std_height',
            'cross_mean_width',
            'cross_std_width',
        ]
        feat_names.extend(quarter_feats_names)

        num_quarter_feats = len(quarter_feats_names)

        quarter_arrays = [] # to collect feats for each signal quarter

        for signal_slice_num in range(num_slices):

            # Feature array
            quarter_feats_array = np.zeros(shape=(signals.shape[0], num_quarter_feats))

            for i, signal in tqdm.tqdm(enumerate(
                    signals[:,signal_slice_num*slice_size:(signal_slice_num+1)*slice_size]
            ), total=signals.shape[0]):

                # Extract peak properties
                peak_heights, peak_widths = fps(
                    signal=signal,
                    min_height=2,
                    max_height=20,
                    ratio_range=0.25,
                    max_distance=30,
                    clean_distance=500,
                    rel_height=0.1
                )

                # Compute feats
                feat_list = [
                    peak_heights.size,

                    np.mean(peak_heights) if len(peak_heights) != 0 else np.nan,
                    np.std(peak_heights) if len(peak_heights) != 0 else np.nan,

                    np.mean(peak_widths) if len(peak_widths) != 0 else np.nan,
                    # np.max(peak_widths) if len(peak_widths) != 0 else np.nan,
                    np.std(peak_widths) if len(peak_widths) != 0 else np.nan,
                ]

                for k, feat in enumerate(feat_list):
                    quarter_feats_array[i, k] = feat

                quarter_arrays.append(quarter_feats_array)

        # Compute std across quarters for all base features

        quarter_feats = np.hstack(quarter_arrays)

        cross_quarter_feats_array = np.zeros(shape=(signals.shape[0], num_quarter_feats))

        for col in range(num_quarter_feats):
            cross_quarter_feats_array[:, col] = np.nanstd(
                a=quarter_feats[:, col::num_quarter_feats],
                axis=1,
            )

        feat_arrays.append(cross_quarter_feats_array)

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


dataset = 'pp_test_db20'
st = time.time()

compute_feats_template = {
    'base-feats': bool(0),
    'quarter-feats': bool(0),
}

feats_to_gen = {
    'base-feats': 'base-feats_v13_v2',
    # 'quarter-feats': 'quarter-feats_v12',
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

