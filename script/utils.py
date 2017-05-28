# -*-coding:utf-8-*-

from collections import defaultdict

import numpy as np
import pandas as pd
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
