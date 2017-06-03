import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import time

os.sys.path.append("/mnt/trident/xiaolan/python/Contest/penguin_click/script")
os.chdir('/mnt/trident/xiaolan/python/Contest/penguin_click/script')


all_data = joblib.load('../processed/all_data_p2')
instance_id = joblib.load('../processed/instance_id')

cvrt_value = all_data['label']
day_values = all_data['click_day']

drop_list = set(["diff_"+str(i)+"_category" for i in range(31)]) - set(["diff_"+str(i)+"_category" for i in [27,28,20,21,22,25]])
xgb_feature = list(set(all_data.columns) - set(["clickTime", "conversionTime", "label", "click_min", "click_day","telecomsOperator","exptv_residence","marriageStatus","appPlatform","userID"]) - drop_list)


def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)
    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)


nn = all_data.shape[0]
np.random.seed(999)
sample_idx = np.random.random_integers(0, 3, nn)
n_trees = 4100
predv_xgb = 0
batch = 0
day_test = 31



# XGB with 4 folds

output_logloss = {}
pred_dict = {}

for idx in [0, 1, 2, 3]:
    filter1 = np.logical_and(np.logical_and(day_values >= 17, day_values < day_test),
                             np.logical_and(sample_idx == idx, True))
    filter_v1 = day_values == day_test
    xt1 = all_data.ix[filter1, xgb_feature]
    yt1 = cvrt_value[filter1]

    xv1 = all_data.ix[filter_v1, xgb_feature]
    yv1 = cvrt_value[filter_v1]

    if xt1.shape[0] <= 0 or xt1.shape[0] != yt1.shape[0]:
        print(xt1.shape, yt1.shape)
        raise ValueError('wrong shape!')

    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(xv1)
    watchlist = [(dtrain, 'train')]
    print(xt1.shape, yt1.shape)
    plst = list(xgb_param.items()) + [('eval_metric', 'logloss')]
    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist, early_stopping_rounds=50)
    batch += 1
    current_pred = xgb1.predict(dvalid)
    yt_hat = xgb1.predict(dtrain)

    pred_dict[idx] = current_pred
    predv_xgb += current_pred

    output_logloss[idx] = logloss(yt_hat, yt1)

    print(logloss(yt_hat, yt1))

    # print('-' * 30, batch, logloss(predv_xgb / batch, yv1))






def generate_pred_with_validation(all_data, xgb_param, xgb_feature, n_trees, day_test=31):
    filter1 = np.logical_and(day_values >= 17, day_values < day_test)
    filter_v1 = day_values == day_test

    xt1 = all_data.ix[filter1, xgb_feature]
    yt1 = cvrt_value[filter1]

    xv1 = all_data.ix[filter_v1, xgb_feature]
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
    print('-' * 30, logloss(xgb1.predict(dvalid), cvrt_value[filter_v1]))


def generate_pred_output(all_data, xgb_param, xgb_feature, n_trees, day_test=31):
    filter1 = np.logical_and(day_values >= 17, day_values < day_test)
    filter_v1 = day_values == day_test

    xt1, yt1 = all_data.ix[filter1, xgb_feature], cvrt_value[filter1]
    xv1 = all_data.ix[filter_v1, xgb_feature]

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


n_trees = 4100


xgb_param = {
    'objective': 'binary:logistic',
    "booster": "gbtree",
    "eval_metric": "logloss",
    "tree_method": 'auto',
    "silent": 1,
    "eta": 0.0175,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.5,
    "gamma": 0,
    "seed": 999
}

output = generate_pred_output(all_data, xgb_param,xgb_feature, n_trees, day_test=31)


def pred_with_different_seed(xgb_param):

    num_seed = 4
    xgb_param_dict = {i: xgb_param.copy() for i in range(num_seed)}
    init_seed = 999
    for i in range(4):
        xgb_param_dict[i]['seed'] = init_seed
        init_seed -= 111

    pred_output_dict = {}

    for i in range(4):
        pred_output_dict[i] = generate_pred_output(all_data, xgb_param_dict[i], xgb_feature, 230, day_test=31)

    return pred_output_dict



submission_output = pd.DataFrame({"prob": pred_output, "instanceID": instance_id})


submission_output['prob'] = submission_output.mean(axis=1)


def generate_submission(df, instance_id):
    df = pd.concat([pd.DataFrame({"instanceID": instance_id}), pd.DataFrame(df)], axis=1)
    df.columns = ['instanceID', 'prob']
    submission_path = "/mnt/trident/xiaolan/python/Contest/penguin_click/pred_output/" + time.strftime("%Y%m%d",time.gmtime())
    if not os.path.exists(submission_path):
        os.system("mkdir " + submission_path, exist_ok=True)
    df.to_csv(os.path.join(submission_path, 'submission', time.strftime("_%H%M", time.gmtime()), '.csv'), index=False)


generate_submission(output, instance_id)
