import numpy as np
import pandas as pd
import time, datetime, os, glob, re
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Permute
from keras.callbacks import EarlyStopping, TensorBoard
from keras.losses import binary_crossentropy
from keras import optimizers, backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from utils import plot_aux_visu, save_submission, plot_batch_1dcnn


class CNNModel:
    '''
    1D CNN Class definition
    '''

    def internal_preprocessing(self, train_tensor, test_tensor, augment_train=None):
        '''
        Normalize (mean and variance) and replace nans with zeros in raw train and test tensors

        :param train_tensor: b x t x c numpy array, where b = num signals, t = num intervals,
            c = num features per interval,
        :param test_tensor: see train tensor
        :param augment_train: Float, >= 1. If true, will add
            np.round('augment_train' * num entries) new entries
            to train set, by randomly rolling signals. Defaults to
            None, no augmentation.
        :return: tuple (train, test), where 'train' and 'test' are the normalized and preprocessed input tensors
        '''

        print(f'>   CNN : Starting internal preprocessing . . .')
        # Unroll tensors, normalize, roll back

        # Put channels (features) first
        train_tensor = np.swapaxes(train_tensor, 0, 2)
        test_tensor = np.swapaxes(test_tensor, 0, 2)

        # Keep shape backup to roll back later after norm
        train_shape, test_shape = train_tensor.shape, test_tensor.shape

        # Unroll
        train_tensor = np.reshape(train_tensor, newshape=(train_shape[0], -1))
        test_tensor = np.reshape(test_tensor, newshape=(test_shape[0], -1))

        # Normalize
        train_mean = np.nanmean(train_tensor, axis=1)[:, None]
        train_std = np.nanstd(train_tensor, axis=1)[:, None]
        train_tensor = (train_tensor - train_mean) / train_std
        test_tensor = (test_tensor - train_mean) / train_std

        # Roll back and undo axis swap
        train_tensor = np.reshape(train_tensor, newshape=train_shape).swapaxes(0, 2)
        test_tensor = np.reshape(test_tensor, newshape=test_shape).swapaxes(0, 2)

        # Replace nans
        train_tensor[np.isnan(train_tensor)] = 0
        test_tensor[np.isnan(test_tensor)] = 0

        # Perform optional data augmentation
        if augment_train is not None:
            np.random.seed(1)

            # Number of new signals to add
            num_new_signals = np.round(augment_train * train_tensor.shape[0]).astype(int)

            # Random roll shifts
            quarter = int(train_tensor.shape[1]/4)
            shifts = np.random.randint(low=quarter, high=3*quarter, size=num_new_signals)
            # Random indexes to augment
            num_signals = train_tensor.shape[0]
            rand_ixs = np.random.randint(low=0, high=num_signals-1, size=num_new_signals)

            # Generate augmented data
            extra_train_entries = np.roll(train_tensor[rand_ixs], shift=shifts, axis=1)

            # Append to train data
            train_tensor = np.concatenate((train_tensor, extra_train_entries), axis=0)

            # Also update target data
            self.y_tgt = np.concatenate((self.y_tgt, self.y_tgt[rand_ixs]))

        print(f'>   CNN : Finished internal preprocessing.')
        return train_tensor, test_tensor

    # Constructor
    def __init__(self, train_tensor_path, test_tensor_path, y_tgt, sample_weight, output_dir,
                 default_threshold, optimize_threshold, model_name, augment_train=None):

        # Input control
        if train_tensor_path is None or test_tensor_path is None:
            raise ValueError('Error initializing CNN - must provide train and test sets')

        # Setup target
        self.y_tgt = y_tgt

        # Load input tensors
        self.train_tensor = np.load(train_tensor_path)
        self.test_tensor = np.load(test_tensor_path)

        # Temporary correction - put channel last. Preferred option for Keras layers
        self.train_tensor = np.swapaxes(self.train_tensor, 1, 2)
        self.test_tensor = np.swapaxes(self.test_tensor, 1, 2)

        # Internal preprocessing
        self.train_tensor, self.test_tensor = self.internal_preprocessing(
            self.train_tensor, self.test_tensor, augment_train
        )

        # Setup other params
        self.model_name = model_name
        self.output_dir = output_dir
        self.default_threshold = default_threshold
        self.optimize_threshold = optimize_threshold
        self.PRED_BATCH_SIZE = 1000

        # Initialize sample weight
        self.sample_weight = np.ones(shape=self.y_tgt.shape)
        self.sample_weight[self.y_tgt == 1] = sample_weight

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
        self.models.extend([
            load_model(
                filepath=os.getcwd() + '/' + models_rel_dir + f'/fold{i}__' + n.split('__')[-1],
                custom_objects={'mcc_keras' : self.mcc_keras}
            ) for i, n in enumerate(nn_names)
        ])

    def mcc_keras(self, y_true, y_pred):
        '''
        Keras MCC metric
        '''
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        tp = K.sum(y_pos * y_pred_pos)
        tn = K.sum(y_neg * y_pred_neg)

        fp = K.sum(y_neg * y_pred_pos)
        fn = K.sum(y_pos * y_pred_neg)

        numerator = (tp * tn - fp * fn)
        denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        return numerator / (denominator + K.epsilon())

    def build_model(self):
        # model layers
        layers = [
            Conv1D(
                input_shape=(self.train_tensor.shape[1], self.train_tensor.shape[2]),
                filters=80,
                kernel_size=10,
                strides=1,
                activation='tanh',
            ),
            # MaxPooling1D(
            #     pool_size=2,
            # ),
            # BatchNormalization(),
            # Conv1D(
            #     filters=120,
            #     kernel_size=5,
            #     strides=1,
            #     activation='tanh',
            # ),
            # MaxPooling1D(
            #     pool_size=4,
            # ),
            # BatchNormalization(),
            # Conv1D(
            #     filters=120,
            #     kernel_size=3,
            #     strides=1,
            #     activation='tanh',
            # ),
            # Dropout(rate=0.25),
            GlobalMaxPooling1D(),
            BatchNormalization(),
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
        y_oof = np.zeros(self.y_tgt.shape)[:, None]

        # Setup stratified CV, by number of phases with PD - hence the np.sum(self.y_tgt)
        num_folds = 5
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(np.arange(0, self.train_tensor.shape[0]))):
            print(f'>   CNN : Fitting fold number {i} . . .')

            '''
            Model setup
            '''

            # Instantiate model
            cnn = self.build_model()

            # Compile model
            cnn.compile(
                optimizer=optimizers.SGD(lr=params['lr'], momentum=0, decay=0, nesterov=False),
                loss=binary_crossentropy,
                metrics=[self.mcc_keras],
            )

            '''
            Model fit
            '''

            # Fit overall definitions
            num_epochs = params['num_epochs']

            print(cnn.summary())
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
            tensorboard = TensorBoard(log_dir=f'../logs/{self.model_name}_{timestamp}')

            hist = cnn.fit(
                x=self.train_tensor[_train],
                y=self.y_tgt[_train],
                epochs=num_epochs,
                batch_size=params['batch_size'],
                validation_data=(self.train_tensor[_eval], self.y_tgt[_eval]),
                sample_weight=self.sample_weight[_train],
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        min_delta=0,
                        patience=params['patience'],
                        mode='min',
                    ),
                    tensorboard,
                ],
                verbose=params['verbose'],
            )

            self.models.append(cnn)

            y_oof[_eval] = cnn.predict(x=self.train_tensor[_eval], batch_size=1000)

            # Calc fold oof MCC
            y_oof_thresholded = (y_oof[_eval] >= self.default_threshold).astype(np.uint8) # threshold
            mcc_oof = matthews_corrcoef(self.y_tgt[_eval], y_oof_thresholded)
            self.fold_mccs.append(mcc_oof)
            print(f'>    CNN : Fold MCC : {mcc_oof:.4f}')

        print('>    CNN : CV results : ')
        print(pd.Series(self.fold_mccs).describe())

    def predict(self, iteration_name, predict_test=True, save_preds=True,
                produce_sub=False, save_aux_visu=False):

        if not self.models:
            raise ValueError('Must fit or load models before predicting')

        if produce_sub:
            predict_test = True

        '''
        Setup CV
        '''

        if predict_test:
            y_test = np.zeros(self.test_tensor.shape[0])[:, None]

        # CV cycle collectors
        y_oof = np.zeros(self.train_tensor.shape[0])[:, None]

        # Setup stratified CV by tgt
        num_folds = 5
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(np.arange(0, self.train_tensor.shape[0]))):
            print(f'>   CNN : Predicting fold number {i} . . .')

            # Predict train oof
            y_oof[_eval] = self.models[i].predict(x=self.train_tensor[_eval], batch_size=self.PRED_BATCH_SIZE)

            # Calc fold oof MCC
            y_oof_thresholded = (y_oof[_eval] >= self.default_threshold).astype(np.uint8)  # threshold
            mcc_oof = matthews_corrcoef(self.y_tgt[_eval], y_oof_thresholded)
            self.fold_mccs.append(mcc_oof)
            print(f'>    CNN : Fold MCC : {mcc_oof:.4f}')

            # Test predictions
            if predict_test:
                y_test += self.models[i].predict(x=self.test_tensor, batch_size=self.PRED_BATCH_SIZE) / num_folds

        final_name = f'cnn_1d_{iteration_name}_{np.mean(self.fold_mccs):.4f}'

        if save_preds:
            train_preds_df = pd.DataFrame(data=y_oof[:, None], columns=[final_name])
            train_preds_df.to_hdf(self.output_dir + f'{final_name}_oof.h5', key='w')

            # No sense in saving test without train hence indent
            if predict_test:
                test_preds_df = pd.DataFrame(data=y_test[:, None], columns=[final_name])
                test_preds_df.to_hdf(self.output_dir + f'{final_name}_test.h5', key='w')

        if produce_sub:
            save_submission(
                y_test,
                sub_name=f'../submissions/{final_name}.csv',
                postprocess=True,
                optimize_threshold=self.optimize_threshold,
                default_threshold=self.default_threshold,
            )

        if save_aux_visu:
            if False:
                plot_aux_visu()
            pass
