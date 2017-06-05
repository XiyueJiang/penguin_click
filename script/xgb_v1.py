import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import utils
import time


def generate_pred_with_validation(all_data, xgb_param, xgb_feature, n_trees, day_test=31):
    filter1 = np.logical_and(day_values >= 17, day_values < day_test)
    filter_v1 = day_values == day_test

    xt1 = all_data.ix[filter1, xgb_feature]
    yt1 = cvrt_value[filter1]

    xv1 = all_data.ix[filter_v1, xgb_feature]
    yv1 = cvrt_value[filter_v1]

    if xt1.shape[0] <= 0 or xt1.shape[0] != yt1.shape[0]:
        print(xt1.shape, xv1.shape)
        raise ValueError('wrong shape!')

    dtrain = xgb.DMatrix(xt1, label=yt1)
    dvalid = xgb.DMatrix(xv1, label=yv1)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    print(xt1.shape, yt1.shape)
    plst = list(xgb_param.items()) + [('eval_metric', 'logloss')]
    xgb1 = xgb.train(plst, dtrain, n_trees, watchlist, early_stopping_rounds=50)
    print('-' * 30, utils.logloss(xgb1.predict(dvalid), cvrt_value[filter_v1]))


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
    xgb_model = xgb.train(plst, dtrain, n_trees, watchlist, early_stopping_rounds=50)


    predv_xgb = xgb_model.predict(dvalid)
    print(xv1.shape)
    return predv_xgb, xgb_model


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


if __name__ == '__main__':

    os.sys.path.append("/mnt/trident/xiaolan/python/Contest/penguin_click/script")
    os.chdir('/mnt/trident/xiaolan/python/Contest/penguin_click/script')

    all_data = joblib.load('../processed/all_data_p3')
    instance_id = joblib.load('../processed/instance_id')

    cvrt_value = all_data['label']
    day_values = all_data['click_day']

    drop_list = set(
        ["diff_" + str(i) + "_category" for i in [19, 21, 25, 28, 26, 15, 11, 8, 7, 6, 23, 20, 5, 4, 17, 13, 12]])

    xgb_feature = list(
        set(all_data.columns) - set(["clickTime", "conversionTime", "label", "click_min", "click_day"]) - drop_list)

    xgb_param = {
        'objective': 'binary:logistic',
        "booster": "gbtree",
        "eval_metric": "logloss",
        "tree_method": 'auto',
        "silent": 1,
        "eta": 0.0123,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.95,
        "colsample_bytree": 0.7,
        "gamma": 0.1,
        "seed": 930114
    }

    output, xgb_model = generate_pred_output(all_data, xgb_param, xgb_feature, n_trees=5500, day_test=31)

    submission_output = pd.DataFrame({"prob": output, "instanceID": instance_id})

    utils.generate_submission(output, instance_id)

    # xgb_model.dump_model("../pre_output/"+ "/mnt/trident/xiaolan/python/Contest/penguin_click/pred_output/" + time.strftime("%Y%m%d",time.gmtime())+"xgb_model.txt")
