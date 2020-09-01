import pandas as pd
from collections import OrderedDict
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import log_loss
from sklearn.metrics import r2_score
from .ValidationData import ValidationData

class Trainer:
    def __init__(self, model, folder, error=None):
        self.folder = folder
        self.eval_error = error
        self.model = model

    def make_fold_data(self, data, fold):
        pool = self.folder._get_fold_(data.train, fold)
        pool.test = data.test
        return pool

    def _get_error_(self, T, P):
        Y = T[[self.model.target]]
        if self.eval_error == 'mae':
            error = mae(Y, P)
        elif self.eval_error == 'rmse':
            error = rmse(Y, P)
        elif self.eval_error == 'logloss':
            error = log_loss(Y, P)
        elif self.eval_error == 'r2':
            error = r2_score(Y, P)
        else:
            raise Exception('Unknown error type')
        return error

    def train(self, data):
        self.folder._generate_folds_(data.train)
        folds = self.folder.folds()
        # errors = [] # TODO: implement average error from folds
        P_valids = OrderedDict()
        for i_fold, fold in enumerate(folds):
            print("Fold = %d / %d" % (i_fold + 1, len(folds)))
            XY = self.make_fold_data(data, fold)
            self.model.train_with_valid(XY)
            P_valid = self.model.predict(XY.valid)
            print('Error %s on Fold_%s: %s' % (self.eval_error, i_fold + 1, self._get_error_(XY.valid, P_valid)))
            P_valids[fold] = P_valid
            print()

        P_valid_concated = self._concat_cross_validation_(data.train, P_valids)
        error_val = self._get_error_(data.train[[self.model.target]], P_valid_concated)
        print('Error %s on Train data: %s' % (self.eval_error, error_val))
        return ValidationData(train=P_valid_concated)

    def train_predict(self, data):
        self.folder._generate_folds_(data.train)
        folds = self.folder.folds()
        # errors = [] # TODO: implement average error from folds
        P_valids = OrderedDict()
        P_tests = OrderedDict()
        for i_fold, fold in enumerate(folds):
            print("Fold = %d / %d" % (i_fold + 1, len(folds)))
            XY = self.make_fold_data(data, fold)
            self.model.train_with_valid(XY)
            P_valid = self.model.predict(XY.valid)
            print('Error %s on Fold_%s: %s' % (self.eval_error, i_fold+1, self._get_error_(XY.valid, P_valid)))
            P_valids[fold] = P_valid
            P_test = self.model.predict(XY.test)
            print('Error %s on Test from Fold_%s: %s' % (self.eval_error, i_fold+1, self._get_error_(XY.test, P_test)))
            P_tests[fold] = P_test
            print()

        P_valid_concated = self._concat_cross_validation_(data.train, P_valids)
        error_val = self._get_error_(data.train[[self.model.target]], P_valid_concated)
        print('Error %s on Train data: %s' % (self.eval_error, error_val))

        P_test_concated = self._combine_test_(P_tests)
        error_test = self._get_error_(data.test[[self.model.target]], P_test_concated)
        print('Error %s on Test data: %s' % (self.eval_error, error_test))

        return ValidationData(train=P_valid_concated, test=P_test_concated)

    def _concat_cross_validation_(self, Y, Ps):
        P_list = [value for key, value in Ps.items()]
        P_valid = pd.concat(P_list, axis=0)
        P_valid = P_valid.reindex(Y.index)
        return P_valid

    def _combine_test_(self, Ps):
        P_list = [value for key, value in Ps.items()]
        P_test = pd.concat(P_list, axis=1)
        P_test_average = P_test.mean(axis=1)
        return P_test_average.to_frame()