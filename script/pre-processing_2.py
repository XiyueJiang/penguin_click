# -*-coding:utf-8-*-

import os

import joblib
import numpy as np
import xgbfir
import xgboost as xgb
from bayes_opt import BayesianOptimization


class BaseAlgo(object):

    def fit_predict(self, train, val=None, test=None, **kwa):
        self.fit(train[0], train[1], val[0] if val else None, val[1] if val else None, **kwa)

        if val is None:
            return self.predict(test[0])
        else:
            return self.predict(val[0]), self.predict(test[0])


class Xgb(BaseAlgo):

    default_params = {
        'objective': 'binary:logistic',
        'base_score': 0.16,
        'eval_metric': 'logloss'
    }

    def __init__(self, params, n_iter=400):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter


    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, name=None, xgbfir_tag=0):

        params = self.params.copy()
        params['seed'] = seed


        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

        if X_eval is None:
            watchlist = [(dtrain, 'train')]
        else:
            deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feature_names)
            watchlist = [(deval, 'eval'), (dtrain, 'train')]


        n_iter = self.n_iter

        self.iter = 0
        self.model = xgb.train(params, dtrain, n_iter, watchlist, verbose_eval=10)
        self.model.dump_model('../processed/xgb-%s.dump' % name, with_stats=True)
        self.feature_names = feature_names

        print("    Feature importances: %s" % ', '.join('%s: %d' % t for t in sorted(self.model.get_fscore().items(), key=lambda t: -t[1])))

        if xgbfir_tag:
            xgbfir.saveXgbFI(self.model, feature_names=self.feature_names, OutputXlsxFile='../processed/xgb-%s.xlx' % name)


    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, seed=42):

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_eval, label=y_eval)

        def fun(**kw):
            params = self.params.copy()
            params['seed'] = seed

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            print("Trying %s..." % str(params))

            self.iter = 0

            model = xgb.train(params, dtrain, 10000, [(dtrain, 'train'), (deval, 'eval')], verbose_eval=20, early_stopping_rounds=100)

            print("Score %f at iteration %d" % (model.best_score, model.best_iteration))

            return - model.best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best logloss: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['max']['max_params']))



def main():

    all_data = joblib.load(os.path.join('../processed', 'all_data_p1'))
    all_data.drop(['conversionTime', 'clickTime'], inplace=True, axis=1)

    presets = {
        'xgb-tst': {
            'model': Xgb({
            'max_depth': 7,
            'eta': 0.1,
            'colsample_bytree': 0.5,
            'subsample': 0.95,
            'min_child_weight': 5,
            }, n_iter=400),
            'param_grid': {'colsample_bytree': [0.2, 1.0]},
        }
    }


    train_x = all_data.ix[all_data['click_day'] < 31, :].drop('label', axis=1)
    train_y = all_data.ix[all_data['click_day'] < 31, 'label']

    feature_names = list(all_data.columns)
    feature_names.remove('label')

    print('xgbfi for feature interaction ...')
    presets['xgb-tst']['model'].fit(X_train=train_x, y_train=train_y, feature_names=feature_names, name='xgb-tst', xgbfir_tag=1)












    n_bags = 5

    for bag in range(n_bags):
        print("    Training model %d..." % bag)








if __name__ == '__main__':



