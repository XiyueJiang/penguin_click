import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

all_data_p1 = joblib.load('../processed/all_data_p1')

cvrt_value = all_data_p1['label']
day_values = all_data_p1['click_day']

day_test = 30

xgb_param = {'max_depth': 10, 'eta':.02, 'objective':'binary:logistic', 'verbose':0,
         'subsample': 1.0, 'min_child_weight':50, 'gamma':0,
         'nthread': 16, 'colsample_bytree':.5, 'base_score':0.16, 'seed': 999}


xgb_feature = ['connectionType', 'adID', 'positionID', 'telecomOperator', 'siteID', 'sitesetID', 'positionType', 'site_position'
               'camgaignID', 'creativeID', 'advertiserID', 'appID', 'appPlatform', 'appCategory', 'app_id_platform',
               'userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'click_hour', 'click_min']

nn = all_data_p1.shape[0]
np.random.seed(999)
sample_idx = np.random.random_integers(0, 3, nn)
n_trees = 200
predv_xgb = 0
batch = 0


def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)
    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)


for idx in [0, 1, 2, 3]:
    filter1 = np.logical_and(np.logical_and(day_values >= 17, day_values < day_test),
                             np.logical_and(sample_idx == idx, True))
    filter_v1 = day_values == day_test

    xt1 = all_data_p1.ix[filter1, xgb_feature]
    yt1 = cvrt_value[filter1].astype('float')
    xt1 = xt1.apply(lambda x: x.astype('float'), axis = 0)

    xv1 = all_data_p1.ix[filter_v1, xgb_feature]
    xv1 = xv1.apply(lambda x: x.astype('float'), axis = 0)

    yv1 = cvrt_value[filter_v1].astype('float')


    if xt1.shape[0] <= 0 or xt1.shape[0] != yt1.shape[0]:
        print(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')

    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(xv1, label=yv1)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    print(xt1.shape, yt1.shape)

    plst = list(xgb_param.items()) + [('eval_metric', 'logloss')]

    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist)


    batch += 1
    predv_xgb += xgb1.predict(dvalid)
    print('-' * 30, batch, logloss(predv_xgb / batch, cvrt_value[filter_v1]))
