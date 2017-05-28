import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import time

os.sys.path.append("/mnt/trident/xiaolan/python/Contest/penguin_click/script")
os.chdir('/mnt/trident/xiaolan/python/Contest/penguin_click/script')

import utils


all_data_p1 = joblib.load('../processed/all_data_p1')
instance_id = joblib.load('../processed/instance_id')

cvrt_value = all_data_p1['label']
day_values = all_data_p1['click_day']

day_test = 30

xgb_param = {'max_depth': 7, 'eta':.15, 'objective':'binary:logistic', 'verbose':0,
         'subsample': 1.0, 'min_child_weight':50, 'gamma': 0,
         'nthread': 16, 'colsample_bytree':.5, 'base_score': 0.12, 'seed': 999}


# xgb_feature = ['connectionType', 'adID', 'positionID', 'telecomOperator', 'siteID', 'sitesetID', 'positionType', 'site_position'
#                'camgaignID', 'creativeID', 'advertiserID', 'appID', 'appPlatform', 'appCategory', 'app_id_platform',
#                'userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence', 'click']

xgb_feature = list(set(all_data_p1.columns) - set(["clickTime", "conversionTime", "label", "click_min"]))


nn = all_data_p1.shape[0]
np.random.seed(999)
sample_idx = np.random.random_integers(0, 3, nn)
n_trees = 500
predv_xgb = 0
batch = 0


def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)
    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)


# XGB with sub-sampling
for idx in [0, 1, 2, 3]:
    filter1 = np.logical_and(np.logical_and(day_values >= 17, day_values < day_test),
                             np.logical_and(sample_idx == idx, True))
    filter_v1 = day_values == day_test

    xt1 = all_data_p1.ix[filter1, xgb_feature]
    yt1 = cvrt_value[filter1]

    xv1 = all_data_p1.ix[filter_v1, xgb_feature]
    yv1 = cvrt_value[filter_v1]

    if xt1.shape[0] <= 0 or xt1.shape[0] != yt1.shape[0]:
        print(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')

    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(xv1, label=yv1)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    print(xt1.shape, yt1.shape)
    plst = list(xgb_param.items()) + [('eval_metric', 'logloss')]
    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist, early_stopping_rounds=50)
    batch += 1
    predv_xgb += xgb1.predict(dvalid)

    print('-' * 30, batch, logloss(predv_xgb / batch, cvrt_value[filter_v1]))


def generate_pred_output(all_data_p1, xgb_param, xgb_feature, n_trees, day_test=31):
    filter1 = np.logical_and(day_values >= 17, day_values < day_test)
    filter_v1 = day_values == day_test

    xt1, yt1 = all_data_p1.ix[filter1, xgb_feature], cvrt_value[filter1]
    xv1 = all_data_p1.ix[filter_v1, xgb_feature]

    if xt1.shape[0] <= 0 or xt1.shape[0] != yt1.shape[0]:
        print(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')

    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(xv1)

    watchlist = [(dtrain, 'train')]
    print(xt1.shape, yt1.shape)
    plst = list(xgb_param.items()) + [('eval_metric', 'logloss')]
    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist, early_stopping_rounds=50)
    predv_xgb = xgb1.predict(dvalid)
    print(xv1.shape)
    return predv_xgb

pred_output = generate_pred_output(all_data_p1, xgb_param, xgb_feature, n_trees = 400,day_test=31)

submission_output = pd.DataFrame({"prob": pred_output, "instanceID": instance_id})


def generate_submission(df):
    submission_path = "/mnt/trident/xiaolan/python/Contest/penguin_click/pred_output/" + time.strftime("%Y%m%d",time.gmtime())
    os.system("mkdir " + submission_path)
    df.to_csv(os.path.join(submission_path, 'submission.csv'), index=False)

generate_submission(submission_output)
