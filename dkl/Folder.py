import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from collections import OrderedDict
from .ValidationData import ValidationData

class Folder:
    def __init__(self, n_folds=5, seed=312):
        self.n_folds = n_folds
        self.seed = seed
        self._folds = None
        self.folds_names = ['Fold_%d' % x for x in range(0, n_folds)]

    def _generate_folds_(self, data):
        index = np.array(data.index)
        n = len(index)

        fold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        idx_folds = []
        for i_train, i_valid in fold.split(np.arange(n)):
            idx_folds.append((i_train, i_valid))

        folds = OrderedDict()
        for i, (i_train, i_valid) in enumerate(idx_folds):
            folds['Fold_%d' % (i,)] = (index[i_train], index[i_valid])

        self._folds = folds

    def _get_fold_(self, data, fold):
        idx_train, idx_valid = self._folds[fold]
        train = data.loc[data.index.isin(idx_train)]
        valid = data.loc[data.index.isin(idx_valid)]
        return ValidationData(train=train, valid=valid)

    def folds(self):
        return self._folds.keys()

class FolderStratified:
    def __init__(self, label='Label', n_folds=5, seed=312):
        self.n_folds = n_folds
        self.seed = seed
        self._folds = None
        self.folds_names = ['Fold_%d' % x for x in range(0, n_folds)]
        self.label = label

    def _generate_folds_(self, data):
        index = np.asarray(data.index)
        Y = np.asarray(data[self.label])
        n = len(index)

        fold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        idx_folds = []
        for i_train, i_valid in fold.split(np.arange(n), Y):
            idx_folds.append((i_train, i_valid))

        folds = OrderedDict()
        for i, (i_train, i_valid) in enumerate(idx_folds):
            folds['Fold_%d' % (i,)] = (index[i_train], index[i_valid])

        self._folds = folds

    def _get_fold_(self, data, fold):
        idx_train, idx_valid = self._folds[fold]
        train = data.loc[data.index.isin(idx_train)]
        valid = data.loc[data.index.isin(idx_valid)]
        return ValidationData(train=train, valid=valid)

    def folds(self):
        return self._folds.keys()
