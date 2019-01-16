import numpy as np
import multiprocessing as mp
import tqdm, glob, os, re, gc, time
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

def atomic_worker(args):
    '''
    Preprocessing worker. See preprocess doc for args description
    '''

    chunk_path, mother_wavelet, num_dec_levels, threshold_value_method, num_low_freq_levels,\
    num_high_freq_levels, threshold_method, save_dir_relative_path, preprocessed_name = args

    # 0. load chunk and get chunk suffix
    print(f'> preprocess.py : Starting preprocessing on chunk at {chunk_path} . . .')
    raw_signals = np.load(chunk_path)
    chunk_suffix = re.compile('_\d+.npy').search(chunk_path).group()

    # 1. retrieve DWT coeffs
    print(f'> preprocess.py : Computing DWT for {raw_signals.shape[0]:d} signals . . .')
    coeffs = pywt.wavedec(raw_signals, mother_wavelet, level=num_dec_levels)
    del raw_signals
    gc.collect()

    # 2. calculate level thresholds
    print(f'> preprocess.py : Computing thresholds . . .')
    if threshold_value_method == 'mad':
        const = 1 / 0.6745  # White-noise-related scaling
        thresholds = [
            const * np.mean(np.abs(coeffs_ - np.mean(coeffs_, axis=1, keepdims=True)), axis=1) * np.sqrt(
                2 * np.log(coeffs_.shape[1])) for coeffs_ in coeffs
        ]
    else:
        raise ValueError(f'Unknown threshold_value_method : {threshold_value_method}')

    # 3. nullify low freq coeffs
    for lvl in range(num_low_freq_levels):
        coeffs[lvl] *= 0

    # 4. apply thresholding operation to high freq coeffs
    print(f'> preprocess.py : Applying thresholds . . .')
    num_coeffs = len(coeffs)
    if num_high_freq_levels == None:
        num_high_freq_levels = num_coeffs - num_low_freq_levels

    for lvl in np.arange(num_coeffs - num_high_freq_levels, num_coeffs, 1):
        coeffs[lvl] = threshold(coeffs[lvl], thresholds[lvl], mode=threshold_method)

    # 5. rebuild signals
    print(f'> preprocess.py : Reconstructing signals . . .')
    pp_signals = pywt.waverec(coeffs, mother_wavelet).astype(np.int8)

    # 6. save rebuilt preprocessed signals
    print(f'> preprocess.py : Saving preprocessed signals . . .')
    np.save(f'{save_dir_relative_path}/{preprocessed_name}/{preprocessed_name}{chunk_suffix}', pp_signals)

    del pp_signals
    gc.collect()
    print(f'> preprocess.py : Finished batch at {chunk_path} .')

def preprocess(
        raw_chunked_signal_relative_path,
        mother_wavelet,
        num_dec_levels,
        threshold_value_method,
        threshold_method,
        num_low_freq_levels,
        num_high_freq_levels,
        save_dir_relative_path,
        preprocessed_name,
        num_workers,
):
    '''
    Preprocess raw signal data - denoising
    
    :param raw_chunked_signal_relative_path: relative path to chunked raw signal directory to preprocess each chunk
    :param mother_wavelet: wavelet family for the DWT
    :param num_dec_levels: num of levels for the DWT
    :param threshold_value_method: method to calculate threshold values per level. Can be one of the
        following : 'mad' (mean absolute deviation)
    :param threshold_method: denoising thresholding method. Can be one of the modes available in pywt
    :param num_low_freq_levels: number of low frequency levels to set to zero ie. remove low large scales
    :param num_high_freq_levels: number of high frequency levels to threshold ie. denoise. If None, all high
        freq levels until 'num_low_freq_levels' will be considered
    :param save_dir_relative_path: relative path to directory where preprocessed signals will be saved
    :param preprocessed_name: name for preprocessed files
    :param num_workers: number of workers for multiprocess. If None, will equal mp.cpu_count()
    :return:
    '''

    st = time.time()

    # load chunked raw signal paths
    print(f'> preprocess.py : Loading raw signal chunk paths . . .')
    raw_chunk_paths = glob.glob(raw_chunked_signal_relative_path + '/*')

    # create directory to save preprocessed chunks if non-existent
    if not os.path.exists(f'../preprocessed_data/{preprocessed_name}'):
        os.mkdir(f'../preprocessed_data/{preprocessed_name}')

    # setup args for atomic workers
    atomic_args = []
    for chunk_path in raw_chunk_paths:
        atomic_args.append(
            (
                chunk_path,
                mother_wavelet,
                num_dec_levels,
                threshold_value_method,
                num_low_freq_levels,
                num_high_freq_levels,
                threshold_method,
                save_dir_relative_path,
                preprocessed_name,
            )
        )

    # distribute to workers
    print(f'> preprocess.py : Starting batch multiprocessing . . .')
    num_workers = mp.cpu_count() if num_workers is None else num_workers
    pool = mp.Pool(processes=num_workers)
    pool.map(atomic_worker, atomic_args)
    pool.close()
    pool.join()

    print(f'> preprocess.py : Done, wall time : {(time.time()-st):.2f} seconds .')

if __name__ == '__main__':

    dataset = 'test'

    preprocess(
        raw_chunked_signal_relative_path=f'../data/{dataset}_chunks',
        mother_wavelet='db20',
        num_dec_levels=None,
        threshold_value_method='mad',
        threshold_method='hard',
        num_low_freq_levels=4,
        num_high_freq_levels=None,
        save_dir_relative_path='../preprocessed_data',
        preprocessed_name=f'pp_{dataset}_db20',
        num_workers=4,
    )