from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
import xgboost as xgb
import numpy as np
import joblib
import os


def xgbCv(train, features, eta):
    # prepare xgb parameters
    params = {
        'objective': 'binary:logistic',
        "booster": "gbtree",
        "eval_metric": "logloss",
        "tree_method": 'auto',
        "silent": 1,
        "eta": eta,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.95,
        "colsample_bytree": 0.7,
        "gamma": 0.1
    }

    cvScore = kFoldValidation(train, features, params)
    print('CV score: {:.6f}'.format(cvScore))
    return -1.0 * cvScore  # invert the cv score to let bayopt maximize


def bayesOpt(train, features):
    ranges = {
        "eta": (0.01, 0.01)
    }

    # proxy through a lambda to be able to pass train and features
    optFunc = lambda eta: \
        xgbCv(train, features, eta)
    bo = BayesianOptimization(optFunc, ranges)
    bo.maximize(init_points=20, n_iter= 20)
    bestlogloss = round((-1.0 * bo.res['max']['max_val']), 6)

    print("\n Best logloss found: %f" % bestlogloss)
    print("\n Parameters: %s" % bo.res['max']['max_params'])


def kFoldValidation(train, features, xgbParams, target='label'):
    kf = KFold(n_splits=3, shuffle=True)
    fold_score = []
    print("Current period learning rate: %f" % xgbParams['eta'])
    for train_index, cv_index in kf.split(train):
        # split train/validation
        X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[cv_index]
        y_train, y_valid = (train[target].as_matrix()[train_index]), (train[target].as_matrix()[cv_index])
        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(xgbParams, dtrain, 10000, evals=watchlist, early_stopping_rounds=50)

        score = gbm.best_score
        fold_score.append(score)

    return np.mean(fold_score)


# def kFoldValidation(train, features, xgbParams, target='label'):
#     # split train/validation
#
#     r1 = np.random.uniform(0, 1, train.shape[0])
#     train_index = r1 < 0.8
#     eval_index = r1 >= 0.2
#
#     X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[eval_index]
#     y_train, y_valid = (train[target].as_matrix()[train_index]), (train[target].as_matrix()[eval_index])
#     dtrain = xgb.DMatrix(X_train, y_train)
#     dvalid = xgb.DMatrix(X_valid, y_valid)
#
#     watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
#     gbm = xgb.train(xgbParams, dtrain, 1000, evals=watchlist, early_stopping_rounds=100)
#
#     score = gbm.best_score
#
#     return score


# def kFoldValidation(train, features, xgbParams, target='label'):
#
#     sample_idx = np.random.random_integers(0, 3, train.shape[0])
#
#     bag_score = []
#
#     for idx in [0,1,2]:
#         print("Current eta %f" % xgbParams['eta'])
#         train_current = train.ix[sample_idx == idx, :].copy()
#
#         r1 = np.random.uniform(0, 1, train_current.shape[0])
#         train_index = r1 < 0.8
#         eval_index = r1 >= 0.8
#
#         X_train, X_valid = train_current[features].as_matrix()[train_index], train_current[features].as_matrix()[eval_index]
#         y_train, y_valid = (train_current[target].as_matrix()[train_index]), (train_current[target].as_matrix()[eval_index])
#
#         print("train size %d, eval size %d" % (X_train.shape[0],  X_valid.shape[0]))
#         dtrain = xgb.DMatrix(X_train, y_train)
#         dvalid = xgb.DMatrix(X_valid, y_valid)
#
#         watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
#         gbm = xgb.train(xgbParams, dtrain, 5000, evals=watchlist, early_stopping_rounds=50)
#
#         score = gbm.best_score
#         print("iteration %d score: %f5" % (idx, score))
#         bag_score.append(score)
#
#     return np.mean(bag_score)


all_data = joblib.load('../processed/all_data_p3')
instance_id = joblib.load('../processed/instance_id')
train_data = all_data.ix[all_data.click_day < 31]

drop_list = set(["diff_"+str(i)+"_category" for i in [19, 21, 25, 28, 26,15, 11, 8, 7, 6, 23, 20, 5, 4, 17, 13, 12]])
xgb_feature = list(set(all_data.columns) - set(["clickTime", "conversionTime", "label", "click_min", "click_day"]) - drop_list)

bayesOpt(train_data, xgb_feature)


