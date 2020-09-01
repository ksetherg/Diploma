from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, GlobalAveragePooling1D, Concatenate
from keras.layers import ZeroPadding1D, Dropout, BatchNormalization, Activation, Permute, Multiply
from keras.layers import LSTM,  Masking, Reshape, AveragePooling1D, Add, UpSampling1D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import pandas as pd


class FCNet:
    def __init__(self, target, features, weight=None, mode='Regressor', learning_rate=0.01):
        self.model = None
        self.target = target
        self.features = features
        self.mode = mode
        self.weight = weight
        self.learning_rate = learning_rate

    def _set_model_(self):
        if self.mode == 'Regressor':
            input_layer = Input(1)
            x = Dense(16)(input_layer)
            x = BatchNormalization()(x)
            x = Activation(activation='relu')(x)
            # x = Dropout(0.2)(x)

            x = Dense(32)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            # x = Dropout(0.2)(x)

            # x = Dense(4)(x)
            # x = BatchNormalization()(x)
            # x = Activation('relu')(x)
            # x = Dropout(0.2)(x)

            output_layer = Dense(1, activation='linear')(x)
            model = Model(inputs=input_layer, outputs=output_layer)

            model.compile(
                            loss='mse',
                            optimizer = Adam(lr=self.learning_rate), 
                            metrics=['mae']
                            )

            self.early_stopping_metric = 'loss'

        elif self.mode == 'Classifier':
            input_layer = Input(1)
            x = Dense(32)(input_layer)
            x = BatchNormalization()(x)
            x = Activation(LeakyReLU())(x)
            # x = Dropout(0.2)(x)

            # x = Dense(64)(x)
            # x = BatchNormalization()(x)
            # x = Activation(LeakyReLU())(x)
            # x = Dropout(0.2)(x)

            x = Dense(16)(x)
            x = BatchNormalization()(x)
            x = Activation(LeakyReLU())(x)
            # x = Dropout(0.2)(x)

            output_layer = Dense(1)(x)
            x = Activation('softmax')(x)
            model = Model(inputs=input_layer, outputs=output_layer)

            model.compile(
                            loss='binary_crossentropy',
                            optimizer = Adam(lr=self.learning_rate), 
                            metrics=['accuracy']
                            )

            self.early_stopping_metric = 'loss'

        else:
            raise Exception('Unknown mode %s' % self.mode)

        self.model = model

    def train_with_valid(self, XY):
        X_train, Y_train = XY.train[self.features].to_numpy(), XY.train[self.target].to_numpy()
        X_valid, Y_valid = XY.valid[self.features].to_numpy(), XY.valid[self.target].to_numpy()
        print('Training FCN with validation')
        print('X_train = %s Y_train = %s' % (X_train.shape, Y_train.shape))
        print('X_valid = %s Y_valid = %s' % (X_valid.shape, Y_valid.shape))
        print()
        '''training'''
        self._set_model_()
        es = EarlyStopping(monitor=self.early_stopping_metric, patience=20)
        if self.weight is None:
            self.model.fit(X_train, Y_train, epochs=4000, callbacks=[es], validation_data=[X_valid, Y_valid], batch_size=64)
        else:
            weights_train, weights_valid = XY.train[self.weight].to_numpy(), XY.valid[self.weight].to_numpy()
            self.model.fit(X_train, Y_train, sample_weight=weights_train, epochs=4000, callbacks=[es], validation_data=[X_valid, Y_valid, weights_valid], batch_size=32)

    def predict(self, X):
        X = X[self.features]
        if self.model is None:
            raise Exception('Train your model before')
        print('Predicting FCN')
        print('X = %s' % (X.shape,))
        print()
        '''predict'''
        prediction = self.model.predict(X.to_numpy())
        prediction = pd.DataFrame(prediction, index=X.index, columns=[self.target])
        return prediction
