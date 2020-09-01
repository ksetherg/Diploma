import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def draw_distr(X_train, X_test, W=None):
    sns.set()
    plt.figure(figsize=(10, 10))
    sns.distplot(X_train, label='Train', color='green', norm_hist=True, hist_kws={'alpha': 0.3})
    if W is not None:
        sns.distplot(X_train, hist_kws={'weights': W, 'alpha': 0.8}, kde=False, label='Weights', norm_hist=True)
    sns.distplot(X_test, label='Test', color='red', norm_hist=True, hist_kws={'alpha': 0.3})
    plt.legend()
    plt.show()

def draw_data(train, test):
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(train['X'], train['Target'], c='green', marker='s', label='Train')
    plt.scatter(test['X'], test['Target'], c='red', marker='.', label='Test')
    plt.legend(loc='upper left')
    plt.show()


class SyntheticData:
    '''
    Example:
    # sd = SyntheticDataset()
    # sd.create_train(0,1)
    # sd.create_train(0,np.sqrt(2))
    '''
    def __init__(self, f): 
        self.f = f 
    
    def _create_Xy(self, mean, sigma, shape, noize_sigma):
        X = np.random.normal(mean, sigma, shape)
        y = self.f(X)
        
        if not noize_sigma is None:
            y_noize = np.random.normal(0, noize_sigma, shape)
            y = y + y_noize
        df = pd.DataFrame({'X': X, 'Target': y})
        df['W'] = 1 / shape
        
        return df
        
    def create_train(self, mean, sigma, shape=10000, noize_sigma=0.2):
        return self._create_Xy(mean, sigma, shape, noize_sigma)
        
    def create_test(self, mean, sigma, shape=1000, noize_sigma=None):
        return self._create_Xy(mean, sigma, shape, noize_sigma)