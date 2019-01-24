import numpy as np
import pandas as pd
from lgbm import LgbmModel
from utils import prepare_datasets

def main():

    '''
    Load and preprocess data
    '''

    # Select relevant cached features
    train_feats_list = [
        '../features/pp_train_db20_base-feats_v13_v2.h5',
    ]
    test_feats_list = [
        '../features/pp_test_db20_base-feats_v13_v2.h5',
    ]

    train, test, y_tgt = prepare_datasets(train_feats_list, test_feats_list)

    # Select models to train
    controls = {
        'lgbm-models'   : bool(1),
    }

    model_name = 'v13r_v2'

    feat_blacklist = [
    ]

    '''
    LGBM Models
    '''
    if controls['lgbm-models']:

        lgbm_params = {
            'num_leaves' : 8,
            'learning_rate': 0.2,
            'min_child_samples' : 100,
            'n_estimators': 30,
            'reg_alpha': 0,
            'reg_lambda': 3,
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
            save_preds=True,
            produce_sub=False,
            save_imps=True,
            save_aux_visu=False,
        )

if __name__ == '__main__':
    main()