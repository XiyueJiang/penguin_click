# -*-coding:utf-8-*-

import os

import joblib

from script import utils


def xgrfir():

    all_data = joblib.load(os.path.join('../processed', 'all_data_p1'))
    all_data.drop(['conversionTime', 'clickTime'], inplace=True, axis=1)

    presets = {
        'xgb-tst': {
            'model': utils.Xgb({
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

    # high interaction by xgbfir

    print('xgbfi for feature interaction ...')
    presets['xgb-tst']['model'].fit(X_train=train_x, y_train=train_y, feature_names=feature_names, name='xgb-tst', xgbfir_tag=1)


if __name__ == '__main__':
    xgrfir()

