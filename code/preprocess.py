import numpy as np
import multiprocessing, tqdm
import pyarrow.parquet as pq
import pywt

def threshold(coeffs_2d, values_1d, mode):
    '''
    Custom one-level thresholding function for several signals simultaneously
    
    :param coeffs_2d: 2d numpy array of coefficients for a given level
    :param values_1d: 1d numpy array of threshold values
    :param mode: as defined in https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html#thresholding
    :return: 2d numpy array of thresholded coeffs
    '''

    if mode=='hard':
        # Get places where coefficients are under the respective threshold
        mask_under_thresh = coeffs_2d < values_1d[:, None]
        # Zero those coeffs
        coeffs_2d[mask_under_thresh] = 0
    else:
        raise ValueError(f'Unknown {mode} thresholding mode specified')

    return coeffs_2d

def preprocess(
        raw_signal_relative_path,
        mother_wavelet,
        num_dec_levels,
        threshold_value_method,
        threshold_method,
        num_low_freq_levels,
        num_high_freq_levels,
        save_dir_relative_path,
        filename,
        num_workers,
):
    '''
    
    Preprocess raw signal data - denoising
    
    :param raw_signal_relative_path: relative path to raw signal to preprocess
    :param mother_wavelet: wavelet family for the DWT
    :param num_dec_levels: num of levels for the DWT
    :param threshold_value_method: method to calculate threshold values per level. Can be one of the
        following : 'mad' (mean absolute deviation)
    :param threshold_method: denoising thresholding method. Can be one of the modes available in pywt
    :param num_low_freq_levels: number of low frequency levels to set to zero ie. remove low large scales
    :param num_high_freq_levels: number of high frequency levels to threshold ie. denoise. If None, all high
        freq levels until 'num_low_freq_levels' will be considered
    :param save_dir_relative_path: relative path to directory where preprocessed signals will be saved
    :param filename: name for preprocessed files
    :param num_workers: number of workers for multiprocess. If None, will equal mp.cpu_count()
    :return:
    '''

    # 1. load all raw signals into numpy array
    print(f'> preprocess.py : Loading raw signal data . . .')
    raw_signals = pq.read_pandas(raw_signal_relative_path, columns=[str(i) for i in np.arange(0,900,1)]).to_pandas().values.T

    # 2. retrieve DWT coeffs
    print(f'> preprocess.py : Computing DWT for {raw_signals.shape[0]:d} signals . . .')
    coeffs = pywt.wavedec(raw_signals, mother_wavelet, level=num_dec_levels)

    # 3. calculate level thresholds
    print(f'> preprocess.py : Computing thresholds . . .')
    if threshold_value_method == 'mad':
        const = 1/0.6745 # White-noise-related scaling
        thresholds = [
            const * np.mean(np.abs(coeffs_-np.mean(coeffs_, axis=1, keepdims=True)), axis=1) * np.sqrt(2 * np.log(coeffs_.shape[1])) for coeffs_ in coeffs
        ]
    else:
        raise ValueError(f'Unknown threshold_value_method : {threshold_value_method}')

    # 4. nullify low freq coeffs
    for lvl in range(num_low_freq_levels):
        coeffs[lvl] *= 0

    # 5. apply thresholding operation to high freq coeffs
    print(f'> preprocess.py : Applying thresholds . . .')
    num_coeffs =len(coeffs)
    if num_high_freq_levels == None:
        num_high_freq_levels = num_coeffs - num_low_freq_levels

    for lvl in np.arange(num_coeffs - num_high_freq_levels, num_coeffs, 1):
        coeffs[lvl] = threshold(coeffs[lvl], thresholds[lvl], mode=threshold_method)

    # 6. rebuild signals
    print(f'> preprocess.py : Reconstructing signals . . .')
    pp_signals = pywt.waverec(coeffs, mother_wavelet)

    # 7. save rebuilt signals
    print(f'> preprocess.py : Saving preprocessed signals . . .')
    np.save(save_dir_relative_path + '/' + filename + '.npy', pp_signals)
    print(f'> preprocess.py : Done .')

if __name__ == '__main__':
    preprocess(
        raw_signal_relative_path='../data/train.parquet',
        mother_wavelet='db2',
        num_dec_levels=None,
        threshold_value_method='mad',
        threshold_method='hard',
        num_low_freq_levels=4,
        num_high_freq_levels=None,
        save_dir_relative_path='../preprocessed_data',
        filename='pp_demo_db2',
        num_workers=None,
    )