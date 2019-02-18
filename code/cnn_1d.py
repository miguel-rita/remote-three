import numpy as np
import pandas as pd
import time, datetime, os, glob, re

from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.losses import binary_crossentropy
from keras import optimizers, backend as K
from keras.utils import Sequence

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from utils import plot_aux_visu, save_submission

class CNN_Generator(Sequence):
    '''
    Generator to handle data for CNN training. Includes signal normalization
    '''

    def preprocess_signal_batch(self, x_batch, max_abs_height=20):
        '''
        Performs a series of preprocessing steps to the signal batch. Currently:
        1) Remove peaks with abs height > max_abs_height
        2) Scale by the max_abs_height

        :param x_batch: a n-by-m numpy array containing a batch o signals
        :return: preprocessed batch
        '''

        x_batch[np.abs(x_batch) > max_abs_height] = 0
        x_batch = x_batch.astype(np.float16) / max_abs_height
        return x_batch

    def __init__(self, chunks_path, labels, chunks_per_batch, chunk_subset=None):
        '''
        Init generator (train and test compatible)
        :param chunks_path: path to folder containing all preprocessed dataset chunks
        :param labels: np 1d array containg target
        :param chunks_per_batch: define batch size by number of chunks. Must divide num of chunks exactly
        :param chunk_subset: (list, optional) if provided, will only consider chunks with number
            in chunk_subset. Useful to implement cross validation later on
        '''

        # Get sorted paths to signal chunks
        chunk_paths = glob.glob(chunks_path + '/*')  # unsorted paths
        chunk_suffixes = [re.search('_\d+\.npy', chunk_name).group() for chunk_name in chunk_paths]  # get suffixes
        chunk_numbers = np.array([int(suffix[1:-4]) for suffix in chunk_suffixes])  # grab number only from each suffix
        arg_sort = np.argsort(chunk_numbers)  # get correct order

        # If we want only a subset of chunks
        if chunk_subset is not None:
            sub_arg_sort = np.array([arg for arg in arg_sort if arg in chunk_subset])
            self.chunk_paths = [chunk_paths[i] for i in sub_arg_sort]
        else:
            self.chunk_paths = [chunk_paths[i] for i in arg_sort]

        # One batch will contain chunks_per_batch * signals_per_chunk signals
        if len(self.chunk_paths) % chunks_per_batch != 0:
            raise ValueError('CNN_Generator : chunks_per_batch must divide exactly num. of chunks in memory')
        self.chunks_per_batch = chunks_per_batch

        # Total number of batches
        self.num_batches = int(len(self.chunk_paths) / self.chunks_per_batch)

        # Get chunk sizes without loading entire chunks - just headers
        all_chunk_paths = [chunk_paths[i] for i in arg_sort]
        chunk_sizes = [np.load(cp, mmap_mode='r').shape[0] for cp in all_chunk_paths]
        label_ix_range = np.hstack(([0], np.cumsum(chunk_sizes)))

        # Store labels per chunk for all chunks
        if labels is not None:
            self.labels = []
            for i, ix in enumerate(label_ix_range[:-1]):
                self.labels.append(labels[label_ix_range[i]:label_ix_range[i+1]])

            # If we want only a chunk subset remove extra labels
            self.labels = [labels_ for lb_num, labels_ in enumerate(self.labels) if lb_num in chunk_subset]

    def __len__(self):

        # Total number of batches this generator can produce
        return self.num_batches

    def __getitem__(self, idx):

        # Load and return a batch, consisting of self.chunks_per_batch chunks and respective targets (if not None)
        chunk_ix = idx * self.chunks_per_batch
        chunks = [np.load(self.chunk_paths[chunk_ix + offset]) for offset in range(self.chunks_per_batch)]
        x_batch = np.vstack(chunks)

        # Final preprocessing
        x_batch = self.preprocess_signal_batch(x_batch)

        if self.labels is None:
            return x_batch

        y_batch = np.hstack(self.labels[chunk_ix:chunk_ix+self.chunks_per_batch])

        x_batch = np.expand_dims(x_batch, 1)

        return x_batch, y_batch

class CNNModel:
    '''
    CNN Class definition
    '''

    # Constructor
    def __init__(self, train_chunks_path, test_chunks_path, y_tgt, sample_weight, output_dir, default_threshold, optimize_threshold):

        # Input control
        if train_chunks_path is None:
            raise ValueError('Error initializing CNN - must provide at least a training set')
        is_train_only = True if test_chunks_path is None else False

        # Dataset info
        self.train_chunks_path = train_chunks_path
        self.test_chunks_path = test_chunks_path

        # Get train and test sizes without loading chunks - just headers
        train_chunk_sizes = [np.load(cp, mmap_mode='r').shape[0] for cp in glob.glob(train_chunks_path + '/*.npy')]
        self.num_train_samples = np.sum(train_chunk_sizes)
        self.num_train_chunks = len(train_chunk_sizes)
        test_chunk_sizes = [np.load(cp, mmap_mode='r').shape[0] for cp in glob.glob(test_chunks_path + '/*.npy')]
        self.num_test_samples = np.sum(test_chunk_sizes)

        self.y_tgt = y_tgt

        # Other params
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

    def matthews_correlation_keras(self, y_true, y_pred):
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
                filters=100,
                kernel_size=500,
                strides=100,
                input_shape=(1,800000),
                data_format='channels_first'
            ),
            MaxPooling1D(
                pool_size=2,
            ),
            Conv1D(
                filters=100,
                kernel_size=10,
                strides=1,
                activation='relu',
                data_format='channels_first'
            ),
            MaxPooling1D(
                pool_size=10,
            ),
            Conv1D(
                filters=100,
                kernel_size=10,
                strides=1,
                activation='relu',
                data_format='channels_first'
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
        y_oof = np.zeros(self.num_train_samples)

        # Setup stratified CV, by number of phases with PD - hence the np.sum(self.y_tgt)
        num_folds = 5
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(np.arange(0, self.num_train_chunks))):
            print(f'>   CNN : Fitting fold number {i} . . .')

            # Setup fold data generators
            train_generator = CNN_Generator(
                chunks_path=self.train_chunks_path,
                labels=self.y_tgt,
                chunks_per_batch=params['chunks_per_batch'],
                chunk_subset=_train,
            )
            eval_generator = CNN_Generator(
                chunks_path=self.train_chunks_path,
                labels=self.y_tgt,
                chunks_per_batch=params['chunks_per_batch'],
                chunk_subset=_eval,
            )
            pred_generator = CNN_Generator(
                chunks_path=self.train_chunks_path,
                labels=None,
                chunks_per_batch=params['chunks_per_batch'],
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
                metrics=[self.matthews_correlation_keras],
            )

            '''
            Model fit
            '''

            # Fit overall definitions
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

    def predict(self, iteration_name, predict_test=True, save_preds=True,
                produce_sub=False, save_aux_visu=False, pred_chunks_per_batch=10):

        if not self.models:
            raise ValueError('Must fit or load models before predicting')

        if produce_sub:
            predict_test = True

        '''
        Setup CV
        '''

        if predict_test:
            y_test = np.zeros(self.num_test_samples)

        # CV cycle collectors
        y_oof = np.zeros(self.num_train_samples)

        # Setup stratified CV by tgt
        num_folds = 5
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(np.arange(0, self.num_train_chunks))):
            print(f'>   CNN : Predicting fold number {i} . . .')

            # Setup fold generators
            oof_pred_generator = CNN_Generator(
                chunks_path=self.train_chunks_path,
                labels=None,
                chunks_per_batch=pred_chunks_per_batch,
                chunk_subset=_eval,
            )
            test_pred_generator = CNN_Generator(
                chunks_path=self.test_chunks_path,
                labels=None,
                chunks_per_batch=pred_chunks_per_batch,
                chunk_subset=_eval,
            )

            # Predict train oof
            y_oof[_eval] = self.models[i].predict_generator(
                generator=oof_pred_generator,
            )

            # Calc fold oof MCC
            # Calc fold oof MCC
            y_oof_thresholded = (y_oof[_eval] >= self.default_threshold).astype(np.uint8)  # threshold
            mcc_oof = matthews_corrcoef(self.y_tgt[_eval], y_oof_thresholded)
            self.fold_mccs.append(mcc_oof)
            print(f'>    nn : Fold MCC : {mcc_oof:.4f}')

            # Test predictions
            if predict_test:
                y_test += self.models[i].predict_generator(generator=oof_pred_generator) / num_folds

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
                postprocess=False,
                optimize_threshold=self.optimize_threshold,
                default_threshold=self.default_threshold,
            )

        if save_aux_visu:
            if False:
                plot_aux_visu()
            pass
