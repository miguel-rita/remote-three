import numpy as np
import pandas as pd
import tqdm, os, glob, re
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import Sequence

class CNN_Generator(Sequence):
    '''
    Generator to handle data for CNN training. Includes signal normalization
    '''

    def preprocess_signal_batch(self, x_batch, max_abs_height=20):
        '''
        Performs a series of preprocessing steps to the signal batch. Currently:
        1) Remove peaks with abs height > max_abs_height
        2) Scale by the max_abs_height

        :param x_batch: a n-by-m numpy array containing a batch o signals
        :return: preprocessed batch
        '''

        x_batch[np.abs(x_batch) > max_abs_height] = 0
        x_batch = x_batch.astype(np.float16) / max_abs_height
        return x_batch

    def __init__(self, chunks_path, labels, chunks_per_batch, chunk_subset=None):
        '''
        Init generator (train and test compatible)
        :param chunks_path: path to folder containing all preprocessed dataset chunks
        :param labels: np 1d array containg target
        :param chunks_per_batch: define batch size by number of chunks. Must divide num of chunks exactly
        :param chunk_subset: (list, optional) if provided, will only consider chunks with number
            in chunk_subset. Useful to implement cross validation later on
        '''

        # Get sorted paths to signal chunks
        chunk_paths = glob.glob(chunks_path + '/*')  # unsorted paths
        chunk_suffixes = [re.search('_\d+\.npy', chunk_name).group() for chunk_name in chunk_paths]  # get suffixes
        chunk_numbers = np.array([int(suffix[1:-4]) for suffix in chunk_suffixes])  # grab number only from each suffix
        arg_sort = np.argsort(chunk_numbers)  # get correct order

        # If we want only a subset of chunks
        if chunk_subset is not None:
            sub_arg_sort = np.array([arg for arg in arg_sort if arg in chunk_subset])
            self.chunk_paths = [chunk_paths[i] for i in sub_arg_sort]
        else:
            self.chunk_paths = [chunk_paths[i] for i in arg_sort]

        # One batch will contain chunks_per_batch * signals_per_chunk signals
        if len(self.chunk_paths) % chunks_per_batch != 0:
            raise ValueError('CNN_Generator : chunks_per_batch must divide exactly num. of chunks in memory')
        self.chunks_per_batch = chunks_per_batch

        # Total number of batches
        self.num_batches = int(len(self.chunk_paths) / self.chunks_per_batch)

        # Get chunk sizes without loading entire chunks - just headers
        all_chunk_paths = [chunk_paths[i] for i in arg_sort]
        chunk_sizes = [np.load(cp, mmap_mode='r').shape[0] for cp in all_chunk_paths]
        label_ix_range = np.hstack(([0], np.cumsum(chunk_sizes)))

        # Store labels per chunk for all chunks

        if labels is not None:
            self.labels = []
            for i, ix in enumerate(label_ix_range[:-1]):
                self.labels.append(labels[label_ix_range[i]:label_ix_range[i+1]])

            # If we want only a chunk subset remove extra labels
            self.labels = [labels_ for lb_num, labels_ in enumerate(self.labels) if lb_num in chunk_subset]

    def __len__(self):

        # Total number of batches this generator can produce
        return self.num_batches

    def __getitem__(self, idx):

        # Load and return a batch, consisting of self.chunks_per_batch chunks and respective targets (if not None)
        chunk_ix = idx * self.chunks_per_batch
        chunks = [np.load(self.chunk_paths[chunk_ix + offset]) for offset in range(self.chunks_per_batch)]
        x_batch = np.vstack(chunks)

        # Final preprocessing
        x_batch = self.preprocess_signal_batch(x_batch)

        if self.labels is None:
            return x_batch

        y_batch = np.hstack(self.labels[idx:idx+self.chunks_per_batch])

        x_batch = np.expand_dims(x_batch, 2)

        return x_batch, y_batch

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

if __name__ == '__main__':
    #parquet_chunker('../data/test.parquet', 200, 'test_chunks')

    gen = CNN_Generator(
        chunks_path='../preprocessed_data/pp_train_db20',
        labels=pd.read_csv('../data/metadata_train.csv')['target'].values,
        chunks_per_batch=4,
        chunk_subset=list(np.arange(8,100))
    )

    first = gen.__getitem__(0)
    third = gen.__getitem__(2)
    print('gen test done')
