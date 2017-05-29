# -*-coding:utf-8-*-

from collections import defaultdict

import numpy as np
import pandas as pd
import xgbfir
import xgboost as xgb
from bayes_opt import BayesianOptimization

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import os


class BaseAlgo(object):

    def fit_predict(self, train, val=None, test=None, **kwa):
        self.fit(train[0], train[1], val[0] if val else None, val[1] if val else None, **kwa)

        if val is None:
            return self.predict(test[0])
        else:
            return self.predict(val[0]), self.predict(test[0])


def batch_generator(X, y=None, batch_size=128, shuffle=False):
    index = np.arange(X.shape[0])

    while True:
        if shuffle:
            np.random.shuffle(index)

        batch_start = 0
        while batch_start < X.shape[0]:
            batch_index = index[batch_start:batch_start + batch_size]
            batch_start += batch_size

            X_batch = X[batch_index, :]

            if sp.issparse(X_batch):
                X_batch = X_batch.toarray()

            if y is None:
                yield X_batch
            else:
                yield X_batch, y[batch_index]


def regularizer(params):
    if 'l1' in params and 'l2' in params:
        return regularizers.l1l2(params['l1'], params['l2'])
    elif 'l1' in params:
        return regularizers.l1(params['l1'])
    elif 'l2' in params:
        return regularizers.l2(params['l2'])
    else:
        return None


def nn_lr(input_shape, params):
    model = Sequential()
    model.add(Dense(1, input_shape=input_shape))

    return model


def nn_mlp(input_shape, params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

        model.add(PReLU())

    model.add(Dense(1, init='he_normal'))

    return model


def nn_mlp_2(input_shape, params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        model.add(PReLU())

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

    model.add(Dense(1, init='he_normal'))

    return model


class Keras(BaseAlgo):

    def __init__(self, arch, params, scale=True, loss='logloss', checkpoint=False):
        self.arch = arch
        self.params = params
        self.scale = scale
        self.loss = loss
        self.checkpoint = checkpoint

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        params = self.params

        if callable(params):
            params = params()

        np.random.seed(seed * 11 + 137)

        if self.scale:
            self.scaler = StandardScaler(with_mean=False)

            X_train = self.scaler.fit_transform(X_train)

            if X_eval is not None:
                X_eval = self.scaler.transform(X_eval)

        checkpoint_path = "/tmp/nn-weights-%d.h5" % seed

        self.model = self.arch((X_train.shape[1],), params)
        self.model.compile(optimizer=params.get('optimizer', 'adadelta'), loss=self.loss)

        callbacks = list(params.get('callbacks', []))

        if self.checkpoint:
            callbacks.append(ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=0))

        self.model.fit_generator(
            generator=batch_generator(X_train, y_train, params['batch_size'], True), samples_per_epoch=X_train.shape[0],
            validation_data=batch_generator(X_eval, y_eval, 800) if X_eval is not None else None, nb_val_samples=X_eval.shape[0] if X_eval is not None else None,
            nb_epoch=params['n_epoch'], verbose=1, callbacks=callbacks)

        if self.checkpoint and os.path.isfile(checkpoint_path):
            self.model.load_weights(checkpoint_path)

    def predict(self, X):
        if self.scale:
            X = self.scaler.transform(X)

        return self.model.predict_generator(batch_generator(X, batch_size=800), val_samples=X.shape[0]).reshape((X.shape[0],))


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


def calc_exptv(df, vn_list, mean0):

    mean0 = mean0
    day_expt = defaultdict()
    df_copy0 = df.ix[:, ['click_day', 'label']].copy()
    day1 = 16
    cred_k = 11

    for vn in vn_list:

        print('exptv1', vn)
        df_copy0[vn] = df[vn]

        for day in range(17, 32):

            if day not in day_expt:
                day_expt[day] = {}

            mask_prev = np.logical_and(df_copy0.click_day > day1, df_copy0.click_day <= day)
            mask_target = np.logical_and(df_copy0.click_day != day, df_copy0.click_day < 31)

            df_copy1 = df_copy0.ix[mask_prev, :].copy()

            df_prev = df_copy1.ix[mask_target, :].copy()

            vn_group = df_prev.groupby(vn)
            group_sum = vn_group['label'].aggregate(np.sum)
            group_cnt = vn_group['label'].aggregate(np.size)

            target = df_copy1.ix[~mask_target, vn].copy()
            _sum = group_sum[target].values
            _cnt = group_cnt[target].values

            _cnt[np.isnan(_sum)] = 0
            _sum[np.isnan(_sum)] = 0

            r = dict()
            exp = (_sum + cred_k * mean0) / (_cnt + cred_k)

            r['exp'] = exp
            r['cnt'] = _cnt

            day_expt[day][vn] = r

        df_copy0.drop(vn, inplace=True, axis=1)

    for vn in vn_list:
        vn_exp = 'exptv_' + vn
        df[vn_exp] = np.zeros(df.shape[0])

        for day in range(17, 32):
            df.loc[df.click_day.values == day, vn_exp] = day_expt[day][vn]['exp']


def calc_diff_day_category(df):

    diff_df = []

    for day_diff in range(1, 31):
        filter_prev = np.logical_and(df.diff_day > 0, df.diff_day <= day_diff)
        grp = df[filter_prev].groupby(['userID', 'appCategory', 'click_day'])['install_counts'].aggregate(np.sum).rename('diff_' + str(day_diff) + '_category')
        diff_df.append(grp)

    concat_df = pd.concat(diff_df, axis=1)
    concat_df[pd.isnull(concat_df)] = 0

    concat_df.reset_index(inplace=True)

    return concat_df


def extract_day(time):
    try:
        time = str(int(time))
        if len(time) == 6:
            day = int(time[:2])
        else:
            day = int(time[:1])
    except:
        day = None

    return day


def pair_interaction(df, column_pairs):
    for pair in column_pairs:
        df0 = df.ix[:, pair].copy()
        print('Combine: ', pair)
        pair_c = np.add(df0[pair[0]].astype('str').values, df0[pair[1]].astype('str').values)

        vn = '_'.join(pair)
        df[vn] = pair_c
        df[vn] = df[vn].astype('category').values.codes

def generate_submission(df):
    submission_path = "/mnt/trident/xiaolan/python/Contest/penguin_click/pred_output/" + time.strftime("%Y%m%d",time.gmtime())
    os.system("mkdir " + submission_path)
    df.to_csv(os.path.join(submission_path, 'submission.csv'), index=False)
