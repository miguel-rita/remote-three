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
        '../features/pp_train_db20_base-feats_v16.h5',
    ]
    test_feats_list = [
        '../features/pp_test_db20_base-feats_v16.h5',
    ]

    train, test, y_tgt = prepare_datasets(train_feats_list, test_feats_list)

    # Select models to train
    controls = {
        'lgbm-models'   : bool(1),
    }

    model_name = 'v23'

    feat_blacklist = [
        'mean_width_rr0.25_md30_rl0.10',
        'mean_height_rr0.25_md30_rl0.10',
        'std_height_rr0.25_md30_rl0.10',
        # 'std_width_rr0.25_md30_rl0.10',
        # 'percen_width_90_rr0.25_md30_rl0.10',
        # 'percen_width_10_rr0.25_md30_rl0.10',
    ]

    '''
    LGBM Models
    '''
    if controls['lgbm-models']:

        lgbm_params = {
            'num_leaves' : 4,
            'learning_rate': 0.4,
            'min_child_samples' : 100,
            'n_estimators': 15,
            'reg_alpha': 0,
            'reg_lambda': 5,
            'bagging_fraction' : 0.7,
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
            sample_weight=1.5,
            default_threshold=0.5,
            optimize_threshold=True,
            postprocess_sub=True,
            feat_blacklist=feat_blacklist,
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