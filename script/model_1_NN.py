# -*-coding:utf-8-*-

import os

import joblib
import numpy as np
import pandas as pd

import utils


def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)
    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)


def main():

    presets = {
        'xgb-tst': {
            'model': utils.Xgb({
                'max_depth': 7,
                'eta': 0.15,
                'colsample_bytree': 0.5,
                'min_child_weight': 5,
                'early_stopping_rounds': 50,
            }, n_iter=1200),
            'param_grid': {'colsample_bytree': [0.2, 1.0]}
        },
        'nn-tst': {
            'model': utils.Keras(utils.nn_mlp, {'l1': 1e-3, 'l2': 1e-3, 'n_epoch': 1, 'batch_size': 128, 'layers': [10]}),
        },
    }

    all_data = joblib.load(os.path.join('../processed', 'all_data_p2'))
    all_data.drop(['conversionTime', 'clickTime'], inplace=True, axis=1)

    train_x = all_data.ix[all_data['click_day'] < 30, :].drop('label', axis=1)
    train_y = all_data.ix[all_data['click_day'] < 30, 'label']

    eval_x = all_data.ix[all_data['click_day'] == 30, :].drop('label', axis=1)
    eval_y = all_data.ix[all_data['click_day'] == 30, 'label']

    test_x = all_data.ix[all_data['click_day'] == 31, :].drop('label', axis=1)

    n_bags = 1
    n_folds = 4
    nn = train_x.shape[0]
    np.random.seed(999)
    folds = np.random.randint(0, n_folds, nn)

    train_p = np.zeros((eval_x.shape[0], n_bags))
    test_foldavg_p = np.zeros((test_x.shape[0], n_bags * n_folds))
    test_fulltrain_p = np.zeros((test_x.shape[0], n_bags))

    log_loss_list = []

    for fold in range(n_folds):

        print("Training fold %d..." % fold)
        fold_train_x = train_x[folds == fold]
        fold_train_y = train_y[folds == fold]

        fold_eval_x = eval_x
        fold_eval_y = eval_y

        fold_test_x = test_x

        fold_feature_names = list(train_x.columns)

        eval_p = np.zeros((fold_eval_x.shape[0], n_bags))

        for bag in range(n_bags):

            print("    Training model %d..." % bag)

            bag_train_x = fold_train_x
            bag_train_y = fold_train_y

            bag_eval_x = fold_eval_x
            bag_eval_y = fold_eval_y

            bag_test_x = fold_test_x

            pe, pt = presets['nn-tst']['model'].\
                fit_predict(train=(bag_train_x, bag_train_y),
                            val=(bag_eval_x, bag_eval_y),
                            test=(bag_test_x,),
                            seed=1314 + 1993 * fold + 114 * bag,
                            feature_names=fold_feature_names,
                            name='%s-fold-%d-%d' % ('nn-tst', fold, bag))

            eval_p[:, bag] += pe
            test_foldavg_p[:, 0 * n_folds * n_bags + fold * n_bags + bag] = pt

            train_p[:, bag] += pe

            print("    log-loss of validation: %.5f" % logloss(pe, fold_eval_y))

        print("  log-loss of mean validation among bag: %.5f" % logloss(np.mean(eval_p, axis=1), fold_eval_y))

        # Calculate err
        log_loss_list.append(logloss(np.mean(eval_p, axis=1), fold_eval_y))

        # Free mem
        del fold_train_x, fold_train_y, fold_eval_x, fold_eval_y

    if True:
        print()
        print("  Full...")

        full_train_x = all_data.ix[all_data['click_day'] < 31, :].drop('label', axis=1)
        full_train_y = all_data.ix[all_data['click_day'] < 31, 'label']

        full_test_x = test_x

        full_feature_names = list(train_x.columns)

        for bag in range(n_bags):

            print("    Training model %d..." % bag)

            bag_train_x = full_train_x
            bag_train_y = full_train_y

            bag_test_x = full_test_x

            pt = presets['nn-tst']['model'].\
                fit_predict(train=(bag_train_x, bag_train_y),
                            test=(bag_test_x,),
                            seed=1314 + 114 * bag,
                            feature_names=full_feature_names,
                            name='%s-full-%d' % ('nn-tst', bag))

            test_fulltrain_p[:, bag] = pt

    # Aggregate predictions
    instance_id = joblib.load(os.path.join('../processed', 'instance_id'))
    train_p = pd.Series(train_p.reshape(train_p.shape[0], ) / n_folds)  # eval
    test_foldavg_p = pd.Series(np.mean(test_foldavg_p, axis=1), index=instance_id.values)  # fold-test-pre
    test_fulltrain_p = pd.Series(np.mean(test_fulltrain_p, axis=1), index=instance_id.values)  # full-test-pre

    # Analyze predictions
    log_loss_mean = np.mean(log_loss_list)
    log_loss_std = np.std(log_loss_list)
    # log_loss = logloss(train_p, train_y)

    print()
    print("CV log loss: %.5f +- %.5f" % (log_loss_mean, log_loss_std))
    # print("CV RES log loss: %.5f" % log_loss)

    print()
    print("Saving predictions... (%s)" % 'nn-tst')

    for part, pred in [('train', train_p), ('test-foldavg', test_foldavg_p), ('test-fulltrain', test_fulltrain_p)]:
        pred.rename('loss', inplace=True)
        pred.index.rename('instance_id', inplace=True)
        pred.to_csv('../preds/%s-%s.csv' % ('nn-tst', part), header=True)


if __name__ == '__main__':
    main()
