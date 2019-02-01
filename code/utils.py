import numpy as np
import pandas as pd
import tqdm, os, re
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

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
    y_test = (y_test >= default_threshold).astype(np.uint8)

    if postprocess:
        y_test = postprocess_submission_vector(y_test)

    # Insert prediction
    sub['target'] = y_test

    # Save back sub
    sub.to_csv(f'../submissions/{sub_name}', index=False)

def postprocess_submission_vector(y_test):

    y_test = np.reshape(y_test, (int(y_test.size/3), -1))

    y_test[np.sum(y_test, axis=1) > 1] = 1
    y_test[np.sum(y_test, axis=1) <= 1] = 0

    y_test = np.reshape(y_test, (-1,))

    return y_test

if __name__ == '__main__':
    parquet_chunker('../data/test.parquet', 200, 'test_chunks')