import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from utils import prepare_datasets

def augment_train_set(train, labels, test, fraction):
    '''Applies semisupervised learning to augment existing train set with
    pseudo labelled test set samples.

    :param train: Pandas Dataframe. Train set.
    :param labels: Numpy 1D array. Train set labels.
    :param test: Pandas Dataframe. Test set.
    :param fraction: Float, from 0 to 1. Percentage of test entries to pseudo label.
    :return: Tuple (Pandas Dataframe, Numpy 1D array). Augmented train set and augmented labels
    '''

    ls = LabelSpreading(alpha=0.01, kernel='rbf', gamma=0.5)

    # Get train and test copies without nans
    train_copy = np.copy(train.values)
    test_copy = np.copy(test.values)
    train_copy[np.isnan(train_copy)] = 0
    test_copy[np.isnan(test_copy)] = 0

    # Pick a fraction of test samples to augment training. Must be in multiples of 3 to keep same phases labelled.
    test_subset = np.full(test.shape[0], False)
    last_test_sample_considered = int(np.round(test.shape[0]/3 * fraction) * 3)
    test_subset[:last_test_sample_considered] = True
    x_test_additional = test_copy[test_subset]

    # Concat picked test sample info to existing train data
    x_train_augmented = np.vstack((train_copy, x_test_additional))
    labels_augmented = np.hstack((labels, np.full(np.sum(test_subset), -1)))

    # Fit pseudo labels
    ls.fit(x_train_augmented, labels_augmented)

    # Build augmented train dataframe, with original nans
    x_train_augmented_with_nans = np.vstack((train.values, test.values[test_subset]))
    train_augmented = pd.DataFrame(data=x_train_augmented_with_nans, columns=train.columns)

    return train_augmented, ls.transduction_

def main():
    # Load desired feature sets
    train_feats_list = [
        '../features/pp_train_db20_base-feats_v21.h5',
    ]
    test_feats_list = [
        '../features/pp_test_db20_base-feats_v21.h5',
    ]

    train, test, y_tgt = prepare_datasets(train_feats_list, test_feats_list)

    fraction = 0.5
    aug_train, aug_y_tgt = augment_train_set(
        train=train,
        labels=y_tgt,
        test=test,
        fraction=fraction,
    )

    # Save augmented train set and target
    aug_train.to_hdf(f'../features/pp_train_db20_base-feats_v21_aug_{fraction:.2f}.h5', key='w')
    np.save(f'../data/y_tgt_aug_{fraction:.2f}.npy', aug_y_tgt)

if __name__ == '__main__':
    main()


