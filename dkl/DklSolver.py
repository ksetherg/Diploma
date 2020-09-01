from scipy import optimize
import numpy as np
import pandas as pd

class Dkl_solver:
    def __init__(self, funcs_list):
        self.funcs_list = funcs_list

    def prepare_data(self, train, test):
        moments = []
        data_matrix = []
        for (key, value) in self.funcs_list:
            whole_mean = pd.concat([train[key], test[key]]).mean()
            # TODO: need to be fill cleverer(median, mean, regression)
            scaled_train = train[key].fillna(0) / whole_mean
            scaled_valid = test[key] / whole_mean
            moments.append(scaled_valid.pow(value).mean())
            data_matrix.append(np.asarray(scaled_train.pow(value)))
        moments = np.asarray(moments)
        X_data = np.asarray(data_matrix)
        Q = np.ones((train.shape[0],)) / train.shape[0]
        return Q, X_data, moments

    """May be it's much easier to save Q, X_data, moments in self"""
    def P_i(self, alpha, Q, X_data):
        """alpha 1d array
        X n x m array"""
        X_lin_comb = np.dot(alpha, X_data)
        Pi = Q * np.exp(X_lin_comb - X_lin_comb.max(initial=0))
        Pi = Pi / Pi.sum()
        return Pi

    def Jacobian(self, alpha, Q, X_data, moments):
        Pi = self.P_i(alpha, Q, X_data)
        Pi_X = X_data * Pi.reshape(1, -1)
        dP_dAlpha = Pi_X - Pi.reshape(1, -1) * Pi_X.sum(axis=1, keepdims=True)
        dP_dAlpha_X = np.dot(dP_dAlpha, X_data.T)
        jac = 2 * np.dot(Pi_X.sum(axis=1) - moments, dP_dAlpha_X)
        return jac

    def opt_func(self, alpha, Q, X_data, moments):
        Pi = self.P_i(alpha, Q, X_data)
        res_func = np.square(np.dot(X_data, Pi) - moments).sum()
        return res_func

    def evaluate_minimization(self, Q, X_data, moments):
        args = (Q, X_data, moments)
        res = optimize.minimize(self.opt_func, args=args, x0=np.zeros(moments.shape), method='BFGS', jac=self.Jacobian)
        print(res)
        P = self.P_i(res.x, Q, X_data)
        print('data efficiency', self.data_prediction_efficiency(P))
        return P

    def make_weights(self, train, test):
        Q, X_data, moments = self.prepare_data(train, test)
        P = self.evaluate_minimization(Q, X_data, moments)
        return P

    def data_prediction_efficiency(self, P):
        return np.exp(-np.multiply(P, np.log(P)).sum()) / P.shape[0]

    def D_kl(self, Q, P):
        """Kullback-Leibler divergence D(Q||P) for discrete distributions
        it is the amount of information lost when P is used to approximate Q"""
        return np.sum(np.multiply(Q, np.log(np.divide(Q, P))))

