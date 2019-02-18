import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from utils import plot_aux_visu, save_importances, save_submission, postprocess_submission_vector

'''
LGBM Class definition
'''

class LgbmModel:

    # Constructor
    def __init__(self, train, test, y_tgt, output_dir, fit_params, sample_weight, default_threshold,
                 optimize_threshold, postprocess_sub, feat_blacklist):

        # dataset
        self.train = train
        self.test = test
        self.y_tgt = y_tgt

        # other params
        self.output_dir = output_dir
        self.fit_params = fit_params
        self.feat_blacklist = feat_blacklist

        self.default_threshold = default_threshold
        self.optimize_threshold = optimize_threshold
        self.postprocess_sub = postprocess_sub

        # Initialize sample weight
        self.sample_weight = np.ones(shape=self.y_tgt.shape)
        self.sample_weight[self.y_tgt == 1] = sample_weight

        # Relabel
        old_tgts = np.reshape(self.y_tgt, (int(self.y_tgt.size/3),-1))
        old_tgts[np.sum(old_tgts, axis=1)>=1] = 1
        self.old_y_tgt = np.copy(self.y_tgt)
        self.y_tgt = np.reshape(old_tgts, (-1,))

    def fit_predict(self, iteration_name, predict_test=True, save_preds=True, produce_sub=False, save_imps=True,
                    save_aux_visu=False):

        if produce_sub:
            predict_test = True

        '''
        Setup CV
        '''

        # CV cycle collectors
        y_oof = np.zeros(self.y_tgt.size)
        if predict_test:
            y_test = np.zeros(self.test.shape[0])
        eval_mccs = []
        imps = pd.DataFrame()

        # Setup stratified CV
        num_folds = 5
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

        # Extract numpy arrays for use in lgbm fit method
        approved_feats = [feat for feat in list(self.train.columns) if feat not in self.feat_blacklist]

        x_all = self.train[approved_feats].values
        if predict_test:
            x_test = self.test[approved_feats].values

        for i, (_train, _eval) in enumerate(folds.split(self.y_tgt, self.y_tgt)):

            print(f'> lgbm : Computing fold number {i} . . .')

            # Setup fold data
            x_train, y_train = x_all[_train], self.y_tgt[_train]
            sample_weight = self.sample_weight[_train]
            x_eval, y_eval = x_all[_eval], self.old_y_tgt[_eval]

            # Setup binary LGBM
            bst = lgb.LGBMClassifier(
                boosting_type='gbdt',
                num_leaves=self.fit_params['num_leaves'],
                learning_rate=self.fit_params['learning_rate'],
                n_estimators=self.fit_params['n_estimators'],
                objective='binary',
                reg_alpha=self.fit_params['reg_alpha'],
                reg_lambda=self.fit_params['reg_lambda'],
                min_child_samples=self.fit_params['min_child_samples'],
                silent=self.fit_params['silent'],
                bagging_fraction=self.fit_params['bagging_fraction'],
                bagging_freq=self.fit_params['bagging_freq'],
                bagging_seed=self.fit_params['bagging_seed'],
                verbose=self.fit_params['verbose'],
            )

            # Train bst
            bst.fit(
                X=x_train,
                y=y_train,
                sample_weight=sample_weight,
                eval_set=[(x_eval, y_eval)],
                eval_names=['\neval_set'],
                early_stopping_rounds=15,
                verbose=self.fit_params['verbose'],
            )

            # Compute and store oof predictions and MCC, performing custom thresholding
            y_oof[_eval] = bst.predict_proba(x_eval)[:, 1]
            y_oof_thresholded = (y_oof[_eval] >= self.default_threshold).astype(np.uint8)
            mcc = matthews_corrcoef(y_eval, y_oof_thresholded)
            eval_mccs.append(mcc)
            print(f'> lgbm : Fold MCC : {mcc:.4f}')

            # Build test predictions
            if predict_test:
                y_test += bst.predict_proba(x_test)[:,1] / num_folds

            # Store importances
            if save_imps:
                imp_df = pd.DataFrame()
                imp_df['feat'] = approved_feats
                imp_df['gain'] = bst.feature_importances_
                imp_df['fold'] = i
                imps = pd.concat([imps, imp_df], axis=0, sort=False)

        print('> lgbm : CV results : ')
        print(pd.Series(eval_mccs).describe())

        '''
        Output wrap-up : save importances, predictions (oof and test), submission and others
        '''

        y_oof_thresholded = (y_oof >= self.default_threshold).astype(np.uint8)
        y_oof_thresholded_pp = postprocess_submission_vector(np.copy(y_oof_thresholded), y_oof)

        final_metric = matthews_corrcoef(self.old_y_tgt, y_oof_thresholded)
        precision_metric = precision_score(self.old_y_tgt, y_oof_thresholded)
        recall_metric = recall_score(self.old_y_tgt, y_oof_thresholded)
        final_metric_pp = matthews_corrcoef(self.old_y_tgt, y_oof_thresholded_pp)
        precision_metric_pp = precision_score(self.old_y_tgt, y_oof_thresholded_pp)
        recall_metric_pp = recall_score(self.old_y_tgt, y_oof_thresholded_pp)
        print(f'> lgbm : MCC for OOF predictions       : {final_metric_pp:.4f} (pp)')
        print(f'> lgbm : Precision for OOF predictions : {precision_metric_pp:.4f} (pp)')
        print(f'> lgbm : Recall for OOF predictions    : {recall_metric_pp:.4f} (pp)\n')
        print(f'> lgbm : MCC for OOF predictions       : {final_metric:.4f} (no pp)')
        print(f'> lgbm : Precision for OOF predictions : {precision_metric:.4f} (no pp)')
        print(f'> lgbm : Recall for OOF predictions    : {recall_metric:.4f} (no pp)')

        if self.postprocess_sub:
            final_name = f'lgbm_{iteration_name}_{final_metric_pp:.4f}_pp'
        else:
            final_name = f'lgbm_{iteration_name}_{final_metric:.4f}'

        if save_imps:
            save_importances(imps, filename_='../importances_gain/imps_' + final_name)

        if save_preds:
            train_preds_df = pd.DataFrame(data=y_oof[:, None], columns=[final_name])
            train_preds_df.to_hdf(self.output_dir + f'{final_name}_oof.h5', key='w')

            # No sense in saving test without train hence indent
            if predict_test:
                test_preds_df = pd.DataFrame(data=y_test[:,None], columns=[final_name])
                test_preds_df.to_hdf(self.output_dir + f'{final_name}_test.h5', key='w')

        if produce_sub:
            save_submission(
                y_test,
                sub_name=f'../submissions/{final_name}.csv',
                postprocess=self.postprocess_sub,
                optimize_threshold=self.optimize_threshold,
                default_threshold=self.default_threshold,
            )

        if save_aux_visu:
            if False:
                plot_aux_visu()
            pass

