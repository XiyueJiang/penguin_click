from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
import xgboost as xgb
import numpy as np

xgb_param = {'max_depth': 7, 'eta':.15, 'objective': 'binary:logistic', 'verbose':0,
         'subsample': 1.0, 'min_child_weight': 3, 'gamma': 0,
         'nthread': 16, 'colsample_bytree':.5, 'base_score': 0.12, 'seed': 999}


def xgbCv(train, features, numRounds, eta, gamma, maxDepth, minChildWeight, colSample):
    # prepare xgb parameters
    params = {
        'objective':'binary:logistic',
        "booster": "gbtree",
        "eval_metric": "logloss",
        "tree_method": 'auto',
        "silent": 1,
        "eta": eta,
        "max_depth": int(maxDepth),
        "min_child_weight": minChildWeight,
        "subsample": 1,
        "colsample_bytree": colSample,
        "gamma": gamma
    }

    cvScore = kFoldValidation(train, features, params, int(numRounds))
    print('CV score: {:.6f}'.format(cvScore))
    return -1.0 * cvScore  # invert the cv score to let bayopt maximize


def bayesOpt(train, features):
    ranges = {
        'numRounds': (200, 2000),
        'eta': (0.05, 0.3),
        'gamma': (0, 25),
        'maxDepth': (5, 10),
        'minChildWeight': (0, 10),
        'colSample': (0, 1)
    }

    # proxy through a lambda to be able to pass train and features
    optFunc = lambda numRounds, eta, gamma, maxDepth, minChildWeight, colSample: \
        xgbCv(train, features, numRounds, eta, gamma, maxDepth, minChildWeight, subsample, colSample)
    bo = BayesianOptimization(optFunc, ranges)
    bo.maximize(init_points=50, n_iter=5, kappa=2, acq="ei", xi=0.0)

    bestlogloss = round((-1.0 * bo.res['max']['max_val']), 6)

    print("\n Best logloss found: %f" % bestlogloss)
    print("\n Parameters: %s" % bo.res['max']['max_params'])


def kFoldValidation(train, features, xgbParams, numRounds,  target='label'):
    kf = KFold(n_splits=4, shuffle=True)
    fold_score = []
    for train_index, cv_index in kf.split(train):
        # split train/validation
        X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[cv_index]
        y_train, y_valid = (train[target].as_matrix()[train_index]), (train[target].as_matrix()[cv_index])
        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(xgbParams, dtrain, numRounds, evals=watchlist, early_stopping_rounds=50)

        score = gbm.best_score
        fold_score.append(score)

    return np.mean(fold_score)




