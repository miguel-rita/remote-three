import numpy as np
import pandas as pd
import time, datetime, os, glob

from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.losses import binary_crossentropy
from keras import optimizers, backend as K

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from utils import plot_aux_visu, save_submission, CNN_Generator

'''
CNN Class definition
'''

class CNNModel:

    # Constructor
    def __init__(self, train_chunks_path, test_chunks_path, y_tgt, sample_weight, output_dir, default_threshold, optimize_threshold):

        # Input control
        if train_chunks_path is None:
            raise ValueError('Error initializing CNN - must provide at least a training set')
        is_train_only = True if test_chunks_path is None else False

        # dataset info
        self.train_chunks_path = train_chunks_path
        self.test_chunks_path = test_chunks_path
        self.num_train_chunks = len(glob.glob(train_chunks_path + '/*.npy'))
        self.num_test_chunks = len(glob.glob(test_chunks_path + '/*.npy'))

        self.y_tgt = y_tgt

        # other params
        self.output_dir = output_dir

        self.default_threshold = default_threshold
        self.optimize_threshold = optimize_threshold

        # Initialize sample weight
        self.sample_weight = np.ones(shape=self.y_tgt.shape)
        self.sample_weight[y_tgt == 1] = sample_weight

        self.models = []
        self.fold_mccs = []

    # State-related methods
    def save(self, model_name):
        '''
        Model save
        '''

        # Setup save dir
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        mean_mcc = np.mean(self.fold_mccs)
        model_name += '__' + timestamp + '__' + f'{mean_mcc:.4f}'
        os.mkdir(os.getcwd() + '/../models/' + model_name)

        # Save model
        for i, nn in enumerate(self.models):
            fold_name = f'fold{i:d}'
            fold_mcc = self.fold_mccs[i]
            filepath = os.getcwd() + '/../models/' + model_name + '/' + fold_name + f'__{fold_mcc:.4f}.h5'
            nn.save(filepath=filepath)

    def load(self, models_rel_dir):
        '''
        Load pretrained nets into memory
        '''
        model_abs_dir = os.getcwd() + '/' + models_rel_dir + '/*.h5'
        nn_names = glob.glob(model_abs_dir)
        nn_names.sort()
        self.models.extend(
            [load_model(os.getcwd() + '/' + models_rel_dir + f'/fold{i}__' + n.split('__')[-1]) for i, n in
             enumerate(nn_names)])

    def build_model(self):


        # model layers
        layers = [
            Conv1D(
                filters=100,
                kernel_size=500,
                strides=100,
                input_shape=(800000,1,),
            ),
            MaxPooling1D(
                pool_size=2,
            ),
            Conv1D(
                filters=100,
                kernel_size=10,
                strides=1,
                activation='relu',
            ),
            MaxPooling1D(
                pool_size=10,
            ),
            Conv1D(
                filters=100,
                kernel_size=10,
                strides=1,
                activation='relu',
            ),
            GlobalAveragePooling1D(
            ),
            Dropout(
                rate=0.5
            ),
            Dense(
                units=1,
                activation='sigmoid',
            )
        ]

        return Sequential(layers=layers)

    # Main methods
    def fit(self, params):

        '''
        Setup CV
        '''

        # CV cycle collectors
        y_oof = np.zeros(self.num_train_chunks)

        # Setup stratified CV, by number of phases with PD - hence the np.sum(self.y_tgt)
        num_folds = 5
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(np.arange(0, self.num_train_chunks))):
            print(f'>   CNN : Fitting fold number {i} . . .')

            # Setup fold data generators
            train_generator = CNN_Generator(
                chunks_path=self.train_chunks_path,
                labels=self.y_tgt,
                chunks_per_batch=4,
                chunk_subset=_train,
            )
            eval_generator = CNN_Generator(
                chunks_path=self.train_chunks_path,
                labels=self.y_tgt,
                chunks_per_batch=4,
                chunk_subset=_eval,
            )
            pred_generator = CNN_Generator(
                chunks_path=self.train_chunks_path,
                labels=None,
                chunks_per_batch=4,
                chunk_subset=_eval,
            )

            '''
            Model setup
            '''

            # Instantiate model
            cnn = self.build_model()

            # Compile model
            cnn.compile(
                optimizer=optimizers.SGD(lr=params['lr'], momentum=0, decay=0, nesterov=False),
                loss=binary_crossentropy,
            )

            '''
            Model fit
            '''

            # Fit overall definitions
            batch_size = params['batch_size']
            num_epochs = params['num_epochs']

            hist = cnn.fit_generator(
                generator=train_generator,
                epochs=num_epochs,
                validation_data=eval_generator,
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=10,
                        mode='min',
                    )
                ],
                verbose=params['verbose'],
            )

            self.models.append(cnn)

            y_oof[_eval] = cnn.predict_generator(
                generator=pred_generator,
            )

            # Calc fold oof MCC
            y_oof_thresholded = (y_oof[_eval] >= self.default_threshold).astype(np.uint8) # threshold
            mcc_oof = matthews_corrcoef(self.y_tgt[_eval], y_oof_thresholded)
            self.fold_mccs.append(mcc_oof)
            print(f'>    nn : Fold MCC : {mcc_oof:.4f}')

        print('>    nn : CV results : ')
        print(pd.Series(self.fold_mccs).describe())

    def predict(self, iteration_name, predict_test=True, save_preds=True, produce_sub=False, save_aux_visu=False):

        if not self.models:
            raise ValueError('Must fit or load models before predicting')

        if produce_sub:
            predict_test = True

        '''
        Setup CV
        '''

        if predict_test:
            y_test = np.zeros((self.x_test.shape[0], self.y_tgt.shape[1]))

        # CV cycle collectors
        y_oof = np.zeros(self.y_tgt.shape)

        # Setup stratified CV, by number of phases with PD - hence the np.sum(self.y_tgt)
        num_folds = 5
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(self.y_tgt[:, 0], np.sum(self.y_tgt, axis=1))):
            print(f'>   CNN : Predicting fold number {i} . . .')

            # Setup fold data
            x_eval, y_eval = self.x_all[_eval], self.y_tgt[_eval]

            # Predict train oof
            y_oof[_eval, :] = self.models[i].predict(x_eval, batch_size=50000)

            # Calc fold oof MCC
            y_oof_1d = np.reshape(y_oof[_eval, :], newshape=(-1,))  # unroll oof preds for MCC
            y_tgt_1d = np.reshape(self.y_tgt[_eval, :], newshape=(-1,))  # unroll tgt for MCC
            y_oof_1d_thresholded = (y_oof_1d >= self.default_threshold).astype(np.uint8)  # threshold
            mcc_oof = matthews_corrcoef(y_tgt_1d, y_oof_1d_thresholded)
            self.fold_mccs.append(mcc_oof)
            print(f'>    nn : Fold MCC : {mcc_oof:.4f}')

            # Test predictions
            if predict_test:
                y_test += self.models[i].predict(self.x_test, batch_size=10000) / num_folds

        final_name = f'mlp_{iteration_name}_{np.mean(self.fold_mccs):.4f}'

        # Reshape train/test preds
        y_oof_1d = np.reshape(y_oof, newshape=(-1,))
        if predict_test:
            y_test_1d = np.reshape(y_test, newshape=(-1,))

        if save_preds:
            train_preds_df = pd.DataFrame(data=y_oof_1d[:, None], columns=[final_name])
            train_preds_df.to_hdf(self.output_dir + f'{final_name}_oof.h5', key='w')

            # No sense in saving test without train hence indent
            if predict_test:
                test_preds_df = pd.DataFrame(data=y_test_1d[:, None], columns=[final_name])
                test_preds_df.to_hdf(self.output_dir + f'{final_name}_test.h5', key='w')

        if produce_sub:
            save_submission(
                y_test_1d,
                sub_name=f'../submissions/{final_name}.csv',
                postprocess=False,
                optimize_threshold=self.optimize_threshold,
                default_threshold=self.default_threshold,
            )

        if save_aux_visu:
            if False:
                plot_aux_visu()
            pass
