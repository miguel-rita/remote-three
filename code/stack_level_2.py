import numpy as np
import pandas as pd
from lgbm import LgbmModel
from utils import prepare_datasets


def main():

    '''
    Load metafeats from level 1
    '''

    # Select relevant cached features
    level_1_train_feats_list = [
        '../level_1_preds/lgbm_v63_0.6879_pp_oof.h5',
        '../level_1_preds/sang_lstm_603_oof.h5',
    ]
    level_1_test_feats_list = [

        '../level_1_preds/lgbm_v63_0.6879_pp_test.h5',
        '../level_1_preds/sang_lstm_603_test.h5',
    ]

    train, test, y_tgt = prepare_datasets(level_1_train_feats_list, level_1_test_feats_list)
    np.save('../other/y_tgt.npy', y_tgt)

    # Select models to train
    controls = {
        'lgbm-models': bool(1),
    }

    model_name = 'stack_v4'

    feat_blacklist = [
    ]

    '''
    LGBM Models
    '''
    if controls['lgbm-models']:

        lgbm_params = {
            'num_leaves' : 2,
            'learning_rate': 0.5,
            'min_child_samples' : 100,
            'n_estimators': 4,
            'reg_alpha': 0,
            'reg_lambda': 5,
            'bagging_fraction' : .7,
            'bagging_freq' : 1,
            'bagging_seed' : 1,
            'silent': 1,
            'verbose': 1,
        }

        lgbm_model_0 = LgbmModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            output_dir='../level_2_preds/',
            fit_params=lgbm_params,
            sample_weight=1,
            default_threshold=0.5,
            optimize_threshold=True,
            postprocess_sub=False,
            feat_blacklist=feat_blacklist,
        )

        lgbm_model_0.fit_predict(
            iteration_name=model_name,
            predict_test=False,
            save_preds=False,
            produce_sub=True,
            save_imps=True,
            save_aux_visu=False,
        )


if __name__ == '__main__':
    main()