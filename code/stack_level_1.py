import numpy as np
import pandas as pd
from lgbm import LgbmModel
from vanilla_mlp import MlpModel
from cnn_1d_v2 import CNNModel
from utils import prepare_datasets

def main():

    '''
    Load and preprocess data
    '''

    # Select relevant cached features
    train_feats_list = [
        # '../features/pp_train_db20_l4h0_.5t_base-feats_v70.h5',
        '../features/pp_train_db20_base-feats_v21.h5',
        # '../features/pp_train_db20_base-feats_v45_bis.h5',
        # '../features/pp_train_db20_nofps-feats_v45.h5',
    ]
    test_feats_list = [
        '../features/pp_test_db20_base-feats_v21.h5',
    ]

    train, test, y_tgt = prepare_datasets(train_feats_list, test_feats_list)

    # Select models to train
    controls = {
        'lgbm-models'   : bool(1),
        'cnn-models'    : bool(0),
        'mlp-models'    : bool(0),
    }

    model_name = 'v64'

    feat_blacklist = [
        # 'percen_width_90_rr0.25_md30_rl0.10',
        # 'std_height_2rr0.25_md30_rl0.10',
        # 'std_width_2rr0.25_md30_rl0.10',
        # 'mean_height_rr0.25_md30_rl0.10',
        # 'percen_width_10_rr0.25_md30_rl0.10',
        # 'percen_width_90_rr0.25_md30_rl0.10',
        # 'std_height_rr0.25_md30_rl0.10_.5t',
        # 'num_peaks_rr0.25_md30_rl0.10_.5t',
        # 'std_width_rr0.25_md30_rl0.10_.5t',
        # 'mean_height_rr0.25_md30_rl0.10_.5t',
        # 'percen_width_10_rr0.25_md30_rl0.10_.5t',
        # 'percen_width_90_rr0.25_md30_rl0.10_.5t',
    ]

    # CNN params

    '''
    LGBM Models
    '''
    if controls['lgbm-models']:
        for seed in range(100):
            sample_weight_jitter = np.random.uniform(-0.2,0.2)
            bag_jitter = np.random.uniform(-0.2,0.1)
            model_name = f'v80_s{seed:d}_j{sample_weight_jitter:.2f}_bagj{bag_jitter:.2f}'
            lgbm_params = {
                'num_leaves' : 2,
                'learning_rate': 0.2,
                'min_child_samples' : 100,
                'n_estimators': 300,
                'reg_alpha': 2,
                'reg_lambda': 10,
                'bagging_fraction' : 0.5+bag_jitter,
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
                sample_weight=1.3+sample_weight_jitter,
                default_threshold=0.5,
                optimize_threshold=True,
                postprocess_sub=True,
                feat_blacklist=feat_blacklist,
                cv_random_seed=seed,
            )

            lgbm_model_0.fit_predict(
                iteration_name=model_name,
                predict_test=True,
                save_preds=True,
                produce_sub=True,
                save_imps=True,
                save_aux_visu=False,
            )

    '''
    CNN models
    '''
    if controls['cnn-models']:

        cnn_output_dir = '../level_1_preds/'

        single_cnn_params = {
            'lr': 0.05,
            # 'dropout_rate': 0,
            'batch_size': 128,
            'num_epochs': 1000,
            'verbose': 1,
            'patience': 10,
        }

        single_cnn_model = CNNModel(
            train_tensor_path='../features_cnn/pp_train_db20_raw-feats_v2.npy',
            test_tensor_path='../features_cnn/pp_test_db20_raw-feats_v2.npy',
            y_tgt=y_tgt,
            sample_weight=1,
            output_dir=cnn_output_dir,
            default_threshold=0.5,
            optimize_threshold=False,
            model_name='raw_v2',
            augment_train=None,
        )

        single_model_name = model_name + '_single_cnn'

        single_cnn_model.fit(params=single_cnn_params)
        # single_cnn_model.save(single_model_name)
        # single_cnn_model.load('../models/v30_single_cnn__2019-02-21_18:43:54__0.4881')

        single_cnn_model.predict(
            iteration_name=single_model_name,
            predict_test=False,
            save_preds=False,
            produce_sub=False,
            save_aux_visu=False,
        )

    '''
    MLP models
    '''
    if controls['mlp-models']:

        mlp_output_dir = '../level_1_preds/'

        # Vanilla MLP

        single_mlp_params = {
            'lr': 0.05,
            'dropout_rate': 0,
            'batch_size': 100,
            'num_epochs': 10000,
            'layer_dims': [20,3],
            'verbose': 0,
        }

        single_mlp_model = MlpModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            sample_weight=1,
            output_dir=mlp_output_dir,
            default_threshold=0.5,
            optimize_threshold=False,
        )

        single_model_name = model_name + '_single'

        # single_mlp_model.fit(params=single_mlp_params)
        # single_mlp_model.save(single_model_name)
        single_mlp_model.load('../models/v23_single__2019-02-02_16:16:22__0.6413')

        single_mlp_model.predict(
            iteration_name=single_model_name,
            predict_test=True,
            save_preds=True,
            produce_sub=True,
            save_aux_visu=False,
        )

if __name__ == '__main__':
    main()