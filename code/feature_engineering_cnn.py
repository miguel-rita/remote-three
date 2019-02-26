'''
Generate custom features from chunked preprocessed signal data
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm, glob, time, pickle, re
from itertools import product
from false_pos_suppression import fps
from utils import get_sorted_chunk_paths
from scipy.signal import find_peaks, peak_widths


def atomic_worker(args):
    chunk_path, compute_feats = args

    # Setup
    feat_names = []
    feat_arrays = []

    # Load preprocessed chunk
    signals = np.load(chunk_path)

    '''
    CNN1D features with FPS
    '''
    if compute_feats['cnn-feats']:

        # Feature names
        segment_feats_names = [
            'num_peaks',
            'mean_height',
            'std_height',
            'mean_width',
            'std_width',
            'percen_height_10',
            'percen_height_50',
            'percen_height_90',
            'percen_width_10',
            'percen_width_50',
            'percen_width_90',
        ]

        # FPS params
        ratio_range = 0.25
        max_distance = 30
        rel_height = 0.1

        suffix = f'_rr{ratio_range:.2f}_md{max_distance:d}_rl{rel_height:.2f}'
        num_segment_feats = len(segment_feats_names)
        num_segments = 1000

        # Feature array
        segment_feats_array = np.zeros(shape=(signals.shape[0], num_segment_feats, num_segments))
        segments = np.array_split(np.arange(0, signals.shape[1]), num_segments)

        # For each segment in each signal
        for i, signal in tqdm.tqdm(enumerate(signals), total=signals.shape[0]):
            for k, segment in enumerate(segments):

                # Extract peak properties
                peak_heights, peak_widths, peak_ixs = fps(
                    signal=signal[segment],
                    min_height=2,
                    max_height=20,
                    ratio_range=ratio_range,
                    max_distance=max_distance,
                    clean_distance=500,
                    rel_height=rel_height,
                )

                # Compute feats
                single_segment_feats = np.array([
                    peak_heights.size,

                    np.mean(peak_heights) if len(peak_heights) != 0 else np.nan,
                    np.std(peak_heights) if len(peak_heights) != 0 else np.nan,

                    np.mean(peak_widths) if len(peak_widths) != 0 else np.nan,
                    np.std(peak_widths) if len(peak_widths) != 0 else np.nan,

                    np.percentile(peak_heights, 10) if len(peak_heights) != 0 else np.nan,
                    np.percentile(peak_heights, 50) if len(peak_heights) != 0 else np.nan,
                    np.percentile(peak_heights, 90) if len(peak_heights) != 0 else np.nan,

                    np.percentile(peak_widths, 10) if len(peak_widths) != 0 else np.nan,
                    np.percentile(peak_widths, 50) if len(peak_widths) != 0 else np.nan,
                    np.percentile(peak_widths, 90) if len(peak_widths) != 0 else np.nan,
                ])

                segment_feats_array[i, :, k] = single_segment_feats

        feat_arrays.append(segment_feats_array)

    '''
    Raw features no FPS, just WT denoising
    '''
    if compute_feats['raw-feats']:

        min_height = 2
        suffix = f'_wt_denoise_minh{min_height:d}'
        num_segment_feats = 14
        num_segments = 5000


        # Feature array
        segment_feats_array = np.zeros(shape=(signals.shape[0], num_segment_feats, num_segments))
        segments = np.array_split(np.arange(0, signals.shape[1]), num_segments)

        # For each segment in each signal
        for i, signal in tqdm.tqdm(enumerate(signals), total=signals.shape[0]):
            for k, segment in enumerate(segments):

                # Extract peak properties
                sub_signal = signal[segment]

                # Get positive and negative peaks
                pos_peak_ixs, pos_peak_props = find_peaks(sub_signal, height=min_height)
                neg_peak_ixs, neg_peak_props = find_peaks(-sub_signal, height=min_height)
                pos_peak_heights = pos_peak_props['peak_heights']
                neg_peak_heights = -neg_peak_props['peak_heights']

                # Combine and sort all peaks
                peak_ixs = np.hstack([pos_peak_ixs, neg_peak_ixs])
                peak_heights = np.hstack([pos_peak_heights, neg_peak_heights])
                sort_order = np.argsort(peak_ixs)
                # peak_ixs = peak_ixs[sort_order]
                peak_heights = peak_heights[sort_order]

                # Compute feats
                basic_feats = np.array([
                    peak_heights.size,
                    np.mean(peak_heights) if len(peak_heights) != 0 else np.nan,
                    np.std(peak_heights) if len(peak_heights) != 0 else np.nan,
                ])

                percentiles = [.01, .01, 1, 5, 20, 50, 80, 95, 99, 99.9, 99.99]
                percentile_feats = np.percentile(peak_heights, percentiles) if len(peak_heights) != 0\
                                       else np.full(len(percentiles), np.nan),

                percentile_feats = np.squeeze(np.array(percentile_feats))
                single_segment_feats = np.hstack((
                    basic_feats,
                    percentile_feats
                ))

                segment_feats_array[i, :, k] = single_segment_feats

        feat_arrays.append(segment_feats_array)

    # Aggregate all feats
    chunk_tensor = np.concatenate(feat_arrays, axis=1) # Axis 1 is the feature axis (0 is signal and 2 segment)

    return chunk_tensor


def gen_feats_cnn(save_rel_dir, save_name, preprocessed_signals_dir, compute_feats):
    '''
    Generate feature tensor from stored preprocessed signal chunks, to
    be used in the 1D CNN.

    :param save_rel_dir (str) Relative dir to save calculated feats
    :param save_name (str) Feat set name
    :param preprocessed_signals_dir (str) Relative dir to preprocessed signals
    :param compute_feats (dict) Dict of bools marking the feats to generate
    :return:
    '''

    np.warnings.filterwarnings('ignore')

    chunk_paths, _ = get_sorted_chunk_paths(preprocessed_signals_dir)

    # Setup atomic args
    atomic_args = [(cp, compute_feats) for cp in chunk_paths]

    print(f'> feature_engineering_cnn : Creating mp pool . . .')

    pool = mp.Pool(processes=mp.cpu_count())
    res = pool.map(atomic_worker, atomic_args)
    pool.close()
    pool.join()

    print(f'> feature_engineering_cnn : Concating and saving results . . .')

    # Concat atomic computed feats and save numpy array
    feat_tensor = np.concatenate(res, axis=0)
    np.save(save_rel_dir + '/' + save_name, feat_tensor)

def main():
    dataset = 'pp_train_db20'
    st = time.time()

    compute_feats_template = {
        'cnn-feats': bool(0),
        'raw-feats': bool(0),
    }

    feats_to_gen = {
        # 'cnn-feats': 'cnn-feats_v1',
        'raw-feats': 'raw-feats_v2',
    }

    for ft_name, file_name in feats_to_gen.items():

        cpt_fts = compute_feats_template.copy()
        cpt_fts[ft_name] = True

        gen_feats_cnn(
            save_rel_dir='../features_cnn',
            save_name=f'{dataset}_{file_name}.npy',
            preprocessed_signals_dir=f'../preprocessed_data/{dataset}',
            compute_feats=cpt_fts,
        )

    print(f'> feature_engineering_cnn : Done, wall time : {(time.time()-st):.2f} seconds .')

if __name__ == '__main__':
    main()
