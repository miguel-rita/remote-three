import numpy as np
import pandas as pd
from lgbm import LgbmModel

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


def main():

    '''
    Load and preprocess data
    '''

    # Select relevant cached features
    train_feats_list = [
        '../features/pp_train_db20_base-feats_v4_jan15.h5',
    ]
    test_feats_list = [
        '../features/pp_test_db20_base-feats_v4_jan15.h5',
    ]

    train, test, y_tgt = prepare_datasets(train_feats_list, test_feats_list)

    # Select models to train
    controls = {
        'lgbm-models'   : bool(1),
    }

    model_name = 'v4_15jan'

    feat_blacklist = []

    '''
    LGBM Models
    '''
    if controls['lgbm-models']:

        lgbm_params = {
            'num_leaves' : 30,
            'learning_rate': 0.1,
            'min_child_samples' : 20,
            'n_estimators': 100,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'bagging_fraction' : 0.8,
            'bagging_freq' : 1,
            'bagging_seed' : 1,
            'silent': 1,
            'verbose': 1,
        }

        lgbm_model_0 = LgbmModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            output_dir='../level_1_preds/',
            fit_params=lgbm_params,
            feat_blacklist=feat_blacklist
        )

        lgbm_model_0.fit_predict(
            iteration_name=model_name,
            predict_test=True,
            save_preds=False,
            produce_sub=True,
            save_imps=True,
            save_aux_visu=False,
        )

if __name__ == '__main__':
    main()