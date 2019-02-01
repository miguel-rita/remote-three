import numpy as np
import pandas as pd
import time, datetime, os, glob

from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras.losses import binary_crossentropy
from keras import optimizers, backend as K

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from utils import plot_aux_visu, save_importances, save_submission, postprocess_submission_vector

'''
MLP Class definition
'''

class MlpModel:

    # Constructor
    def __init__(self, train, test, y_tgt, output_dir, fit_params, sample_weight, default_threshold,
                 optimize_threshold, postprocess_sub, feat_blacklist):

        # Input control
        if train is None:
            raise ValueError('Error initializing MLP - must provide at least a training set')
        is_train_only = True if test is None else False

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

        self.models = []
        self.fold_val_losses = []
        self.fold_mccs = []

        '''
        Input scaling and additional preprocessing for nn
        '''
        # Take care of NAs (derived from 0 peak counts) and standard scale inputs

        self.train = self.train.replace([np.nan], 0)
        self.x_all = self.train.values

        ss = StandardScaler()
        ss.fit(self.x_all)
        self.x_all = ss.transform(self.x_all)

        if not is_train_only:
            self.test = self.test.replace([np.nan], 0)
            self.x_test = self.x_test.values
            self.x_test = ss.transform(self.x_test)

        # Reshape data to learn all 3-phases at once
        self.x_all = np.reshape(self.x_all, newshape=(int(self.x_all.shape[0]/3), -1))
        self.x_test = np.reshape(self.x_test, newshape=(int(self.x_test.shape[0]/3), -1))
        self.y_tgt = np.reshape(self.y_tgt, newshape=(int(self.y_tgt.shape[0]/3), -1))

    # Loss-related methods
    def custom_cross_entropy_loss(self, y_true, y_pred):
        '''
        Multilabel binary CE loss - WIP TODO
        '''

        phase_losses = np.zeros(3)
        for phase in range(3):
            binary_crossentropy(y_true[:,phase], y_pred[:,phase])

        return np.mean(phase_losses)

    # State-related methods
    def save(self, model_name):
        '''
        Model save
        '''

        # Setup save dir
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        mean_loss = np.mean(self.fold_val_losses)
        model_name += '__' + timestamp + '__' + f'{mean_loss:.4f}'
        os.mkdir(os.getcwd() + '/../models/' + model_name)

        # Save model
        for i, nn in enumerate(self.models):
            fold_name = f'fold{i:d}'
            fold_loss = self.fold_val_losses[i]
            filepath = os.getcwd() + '/../models/' + model_name + '/' + fold_name + f'__{fold_loss:.4f}.h5'
            nn.save(filepath=filepath)

    def load(self, models_rel_dir):
        '''
        Load pretrained nets into memory
        '''
        nn_names = glob.glob(os.getcwd() + '/../' + models_rel_dir + '/*.h5')
        nn_names.sort()
        self.models.extend(
            [load_model(os.getcwd() + '/../' + models_rel_dir + f'/fold{i}__' + n.split('__')[-1]) for i, n in
             enumerate(nn_names)])

    def build_model(self, layer_dims, dropout_rate, activation='relu'):

        # create model
        model = Sequential()

        if len(layer_dims) < 1:
            raise ValueError('Mlp must have at least one layer')

        # first layer, smaller dropout
        model.add(Dense(layer_dims[0], input_dim=self.x_all.shape[1], activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # further layers
        for ld in layer_dims[1:]:
            model.add(Dense(ld, activation=activation))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # final layer- sigmoid since multilabel problem
        model.add(Dense(3, activation='sigmoid'))
        return model

    # Main methods
    def fit(self, params):

        '''
        Setup CV
        '''

        # CV cycle collectors
        y_oof = np.zeros(self.y_tgt.shape)

        # Setup stratified CV, by number of phases with PD - hence the np.sum(self.y_tgt)
        num_folds = 5
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(self.y_tgt[:,0], np.sum(self.y_tgt, axis=1))):
            print(f'>   nn : Fitting fold number {i} . . .')

            # Setup fold data
            x_train, y_train = self.x_all[_train], self.y_tgt[_train]
            x_eval, y_eval = self.x_all[_eval], self.y_tgt[_eval]

            '''
            Model setup
            '''

            # Instantiate model
            nn = self.build_model(layer_dims=params['layer_dims'], dropout_rate=params['dropout_rate'],
                                  activation='relu')

            # Compile model
            nn.compile(
                optimizer=optimizers.SGD(lr=params['lr'], momentum=0, decay=0, nesterov=False),
                loss=binary_crossentropy,
            )

            '''
            Model fit
            '''

            # Fit overall definitions
            batch_size = params['batch_size']
            num_epochs = params['num_epochs']

            hist = nn.fit(
                x=x_train,
                y=y_train,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(x_eval, y_eval),
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

            self.models.append(nn)

            y_oof[_eval, :] = nn.predict(x_eval, batch_size=50000)

            # Calc fold oof validation loss
            val_loss = binary_crossentropy(y_eval, y_oof[_eval, :])
            self.fold_val_losses.append(val_loss)
            print(f'>    nn : Fold val loss : {val_loss:.4f}')

            # Calc fold oof MCC
            y_oof_1d = np.reshape(y_oof[_eval, :], newshape=(-1,)) # unroll oof preds for MCC
            y_tgt_1d = np.reshape(self.y_tgt[_eval, :], newshape=(-1,)) # unroll tgt for MCC
            y_oof_1d_thresholded = (y_oof_1d >= self.default_threshold).astype(np.uint8) # threshold
            mcc_oof = matthews_corrcoef(y_tgt_1d, y_oof_1d_thresholded)
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
            y_test = np.zeros((self.test.shape[0], self.weights.size))

        # CV cycle collectors
        y_oof = np.zeros(self.y_tgt.shape)

        # Setup stratified CV, by number of phases with PD - hence the np.sum(self.y_tgt)
        num_folds = 5
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1)

        for i, (_train, _eval) in enumerate(folds.split(self.y_tgt[:, 0], np.sum(self.y_tgt, axis=1))):
            print(f'>   nn : Predicting fold number {i} . . .')

            # Setup fold data
            x_eval, y_eval = self.x_all[_eval], self.y_tgt_oh[_eval]

            # Train predictions (oof)
            y_oof[_eval, :] = self.models[i].predict(x_eval, batch_size=10000)

            # Test predictions
            if predict_test:
                y_test += self.models[i].predict(self.x_test, batch_size=10000) / num_folds

            val_loss = self.weighted_average_crossentropy_numpy(y_eval, y_oof[_eval, :])
            self.fold_val_losses.append(val_loss)
            print(f'>   nn : Fold val loss : {val_loss:.4f}')

        final_name = f'mlp_{iteration_name}_{np.mean(self.fold_val_losses):.4f}'

        if produce_sub:
            save_submission(y_test, sub_name=f'./subs/{final_name}.csv', rs_bins=self.test['rs_bin'].values)

        if save_confusion:
            y_preds = np.argmax(y_oof, axis=1)
            cm = confusion_matrix(self.y_tgt, y_preds, labels=np.unique(self.y_tgt))
            plot_confusion_matrix(cm, classes=[str(c) for c in self.class_codes[np.unique(self.y_tgt)]],
                                  filename_='confusion/confusion_' + final_name, normalize=True)

        if save_preds:

            class_names = [final_name + '__' + str(c) for c in self.class_codes]

            oof_preds = pd.concat([self.train[['object_id']], pd.DataFrame(y_oof, columns=class_names)], axis=1)
            oof_preds.to_hdf(self.output_dir + f'{final_name}_oof.h5', key='w')

            if predict_test:
                test_preds = pd.concat([self.test[['object_id']], pd.DataFrame(y_test, columns=class_names)], axis=1)
                test_preds.to_hdf(self.output_dir + f'{final_name}_test.h5', key='w')

                return oof_preds, test_preds

