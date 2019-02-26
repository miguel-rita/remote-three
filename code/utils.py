import numpy as np
import pandas as pd
import tqdm, os, glob, re
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

def max_streak(vec, gap):
    '''
    Given a vec of indexes, gives the size of the largest interval of consecutive indexes
    not more than gap units apart.
    Eg.
    
    max_streak(vec=[1,10,11,14,20,50], gap=5)
    >> 4
    
    where 4 is in fact the distance between elements at the edge of the largest interval of
    consecutive elements not more than gap distance apart (in this case [10,11,14])
    '''
    best_streak = 0
    curr_streak = 0
    diffs = vec[1:]-vec[:-1]
    for diff in diffs:
        if diff > gap:
            if curr_streak > best_streak:
                best_streak = curr_streak
            curr_streak = 0
        else:
            curr_streak += diff
    if curr_streak > best_streak:
                best_streak = curr_streak
    return best_streak

def parquet_chunker(rel_path_to_file, num_chunks, chunk_name):
    '''
    Utility to split a parquet file at 'rel_path_to_file' into 'num_chunks' numpy .npy chunks
    
    :param rel_path_to_file: Path to target parquet file
    :param num_chunks: Number of .npy chunks to produce
    :param chunk_name: Base name for the produced chunks
    :return: 
    '''

    # get number of signals in pq file
    pq_file_num_columns = pq.ParquetFile(rel_path_to_file).metadata.num_columns

    # get chunk indices
    chunk_ixs = np.array_split(np.arange(pq_file_num_columns), num_chunks)

    # create dir to store chunks
    if not os.path.exists(f'../data/{chunk_name}'):
        os.mkdir(f'../data/{chunk_name}')

    # Add offset of the 8712 training signals if treating test data
    offset = 8712 if rel_path_to_file.find('test') != -1 else 0

    # save each chunk as separate .npy array
    for i, ixs in tqdm.tqdm(enumerate(chunk_ixs), total=len(chunk_ixs)):
        ixs += offset
        chunk = pq.read_pandas(
            rel_path_to_file, columns=[str(ix) for ix in ixs]
        ).to_pandas().values.T.astype(np.int8)
        np.save(f'../data/{chunk_name}/{chunk_name}_{i:d}.npy', chunk)

def prepare_datasets(train_feats_list, test_feats_list):
    '''
    From a list of paths to precomputed features, loads and prepares train, test and target datasets
    for use by the models

    :param train_feats_list: list of relative paths to precomputed train feats
    :param test_feats_list: list of relative paths to precomputed test feats
    :return: tuple of train df, test df, target 1d np array
    '''

    # Concatenate train and test feats
    train_feats_dfs = [pd.read_hdf(path, mode='r') for path in train_feats_list]
    train_feats_df = pd.concat(train_feats_dfs, axis=1)

    test_feats_dfs = [pd.read_hdf(path, mode='r') for path in test_feats_list]
    test_feats_df = pd.concat(test_feats_dfs, axis=1)

    # Read metadata target for train set
    y_target = pd.read_csv('../data/metadata_train.csv')['target'].values

    return train_feats_df, test_feats_df, y_target

def plot_aux_visu():
    # TODO
    pass

def save_importances(imps_, filename_):
    mean_gain = imps_[['gain', 'feat']].groupby('feat').mean().reset_index()
    mean_gain.index.name = 'feat'
    plt.figure(figsize=(6, 17))
    sns.barplot(x='gain', y='feat', data=mean_gain.sort_values('gain', ascending=False))
    plt.title(f'Num. feats = {mean_gain.shape[0]:d}')
    plt.tight_layout()
    plt.savefig(filename_+'.png')
    plt.clf()

def save_submission(y_test, sub_name, postprocess, optimize_threshold, default_threshold):
    # Load template sub
    sub = pd.read_csv('../data/sample_submission.csv')

    # Threshold test predictions
    y_test_probas = np.copy(y_test)
    y_test = (y_test >= default_threshold).astype(np.uint8)

    if postprocess:
        y_test = postprocess_submission_vector(y_test, y_test_probas)

    # Insert prediction
    sub['target'] = y_test

    # Save back sub
    sub.to_csv(f'../submissions/{sub_name}', index=False)

def postprocess_submission_vector(y_test, y_test_probas):

    y_test = np.reshape(y_test, (int(y_test.size/3), -1))
    y_test_probas = np.reshape(y_test_probas, (int(y_test_probas.size/3), -1))

    y_test[np.sum(y_test, axis=1) > 1] = 1
    y_test[(np.sum(y_test, axis=1) == 1) & (np.sum(y_test_probas, axis=1) > 1.25)] = 1
    y_test[(np.sum(y_test, axis=1) == 1) & (np.sum(y_test_probas, axis=1) <= 0.75)] = 0

    y_test = np.reshape(y_test, (-1,))

    return y_test

def get_sorted_chunk_paths(chunks_path):
    '''
    From a path to folder with numpy chunks, return (list of sorted chunk paths, argsort)
    :param chunks_path: See above
    :return: See above
    '''

    chunk_paths = glob.glob(chunks_path + '/*.npy')  # unsorted paths
    chunk_suffixes = [re.search('_\d+\.npy', chunk_name).group() for chunk_name in chunk_paths]  # get suffixes
    chunk_numbers = np.array([int(suffix[1:-4]) for suffix in chunk_suffixes])  # grab number only from each suffix
    arg_sort = np.argsort(chunk_numbers)  # get correct order
    return [chunk_paths[i] for i in arg_sort], arg_sort

def preprocess_1dcnn(chunks_dir, dataset_name, max_abs_height=20):
    '''
    Performs a series of preprocessing steps to all chunks in 'chunks_dir'. Currently:
    1) Remove peaks with abs height > max_abs_height
    2) Scale by the max_abs_height
    Saves preprocessed chunks in ./preprocessed_data_1dcnn.

    :param chunks_dir : String, relative directory to normally preprocessed chunks.
    :param max_abs_height : Float, set to zero every peak greater than 'max_abs_height'.
    :param dataset_name : String, name for the final preprocessed chunks

    :return: --
    '''

    if not os.path.exists(f'../preprocessed_data_1dcnn/{dataset_name}'):
        os.mkdir(f'../preprocessed_data_1dcnn/{dataset_name}')

    chunk_paths, _ = get_sorted_chunk_paths(chunks_dir)

    for i, path in tqdm.tqdm(enumerate(chunk_paths), total=len(chunk_paths)):
        chunk = np.load(path)
        chunk[np.abs(chunk) > max_abs_height] = 0
        chunk = chunk.astype(np.float16) / 20
        np.save(f'../preprocessed_data_1dcnn/{dataset_name}/{dataset_name}_{i:d}.npy', chunk)

def plot_batch_1dcnn(x_batch, y_batch, num_sigs):
    '''
    Visualize one batch of 1dcnn data

    :param x_batch: Numpy array, of shape b x s x t where b = num. of
        signals, s = 1, t = num. of timesteps. Signal data.
    :param y_batch: Numpy array, of shape b where b = num. of signals. Target (binary) data.
    :param num_sigs: Plot the the first 'num_sigs'.
    :return: --
    '''

    x_batch = np.squeeze(x_batch)
    sns.lineplot(x=np.arange(0, x_batch.shape[-1], 100), y=x_batch[0,::100])
    plt.savefig('../visualizations/1dcnn_debug.png')
    plt.clf()
    plt.close()

    print('Visu done.')

if __name__ == '__main__':
    # parquet_chunker('../data/test.parquet', 200, 'test_chunks')
    preprocess_1dcnn(
        chunks_dir='../preprocessed_data/pp_train_db20',
        dataset_name='pp_train_db20_1dcnn',
    )

