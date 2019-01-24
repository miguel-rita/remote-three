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
        # '../features/pp_train_db20_base-feats_v13.h5',
        '../level_1_preds/lgbm_v13r_0.6139_oof.h5',
        '../level_1_preds/lgbm_v13r_v2_0.5567_oof.h5'
    ]
    level_1_test_feats_list = [
        # '../features/pp_test_db20_base-feats_v13.h5',
        '../level_1_preds/lgbm_v13r_0.6139_test.h5',
        '../level_1_preds/lgbm_v13r_v2_0.5567_test.h5'
    ]

    train, test, y_tgt = prepare_datasets(level_1_train_feats_list, level_1_test_feats_list)

    # Select models to train
    controls = {
        'lgbm-models': bool(1),
    }

    model_name = 'stack_v1'

    feat_blacklist = [
    ]

    '''
    LGBM Models
    '''
    if controls['lgbm-models']:
        lgbm_params = {
            'num_leaves' : 2,
            'learning_rate': 0.25,
            'min_child_samples' : 60,
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
            predict_test=False,
            save_preds=False,
            produce_sub=True,
            save_imps=True,
            save_aux_visu=False,
        )


if __name__ == '__main__':
    main()