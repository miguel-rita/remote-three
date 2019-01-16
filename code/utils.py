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

def save_submission(y_test, sub_name):
    # Load template sub
    sub = pd.read_csv('../data/sample_submission.csv')

    # Insert prediction
    sub['target'] = y_test

    # Save back sub
    sub.to_csv(f'../submissions/{sub_name}', index=False)

if __name__ == '__main__':
    parquet_chunker('../data/test.parquet', 200, 'test_chunks')