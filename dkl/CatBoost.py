import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier, Pool


class CatBoost:
    def __init__(self, target, features, weight=None, mode='Regressor', objective='RMSE', logs=True):
        self.model = None
        self.target = target
        self.features = features
        self.mode = mode
        self.weight = weight
        self.logs = logs

        self.model_params = dict(
            thread_count=8,
            iterations=2000,
            loss_function=objective,
            # learning_rate=0.05
        )

        self.training_params = dict(
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=100
        )

    def _set_model_(self):
        if self.mode == 'Regressor':
            self.model = CatBoostRegressor()
            self.model.set_params(**self.model_params)
        elif self.mode == 'Classifier':
            self.model = CatBoostClassifier()
            self.model.set_params(**self.model_params)
        else:
            raise Exception('Unknown mode %s' % self.mode)

    def train_with_valid(self, XY):
        X_train, Y_train = XY.train[self.features], XY.train[self.target]
        X_valid, Y_valid = XY.valid[self.features], XY.valid[self.target]
        if self.weight is None:
            train_pool = Pool(data=X_train, label=Y_train)
            val_pool = Pool(data=X_valid, label=Y_valid)
        else:
            W_train, W_valid = XY.train[self.weight], XY.valid[self.weight]
            train_pool = Pool(data=X_train, label=Y_train, weight=W_train)
            val_pool = Pool(data=X_valid, label=Y_valid, weight=W_valid)
        '''logging'''
        print('Training Model CatBoost with validation')
        print('X_train = %s Y_train = %s' % (X_train.shape, Y_train.shape))
        print('X_valid = %s Y_valid = %s' % (X_valid.shape, Y_valid.shape))
        print()
        '''training'''
        self._set_model_()
        self.model = self.model.fit(train_pool,
                                    eval_set=val_pool,
                                    **self.training_params)
        '''feature importances'''
        if self.logs:
            self._logging_feature_importance_(train_pool)

    def predict(self, X):
        X = X[self.features]
        if self.model is None:
            raise Exception('Train your model before')
        print('Predicting Model CatBoost')
        print('X = %s' % (X.shape,))
        print()
        data_pool = Pool(data=X)
        '''predict'''
        if self.mode == 'Regressor':
            prediction = self.model.predict(data_pool)
        elif self.mode == 'Classifier':
            prediction = self.model.predict(data_pool, prediction_type='Probability')
            prediction = prediction[:, 1]
        prediction = pd.DataFrame(prediction, index=X.index, columns=[self.target])
        return prediction

    def _logging_feature_importance_(self, train_pool):
        if self.model is None:
            raise Exception('Train your model before')
        print('Top features')
        feature_importance = self.model.get_feature_importance(train_pool)
        feature_names = train_pool.get_feature_names()
        for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
            print('{}: {}'.format(name, score))

