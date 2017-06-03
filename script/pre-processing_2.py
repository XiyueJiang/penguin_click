# -*-coding:utf-8-*-

import os

import joblib

from script import utils


def xgrfir():

    all_data = joblib.load(os.path.join('../processed', 'all_data_p3'))
    all_data.drop(['conversionTime', 'clickTime'], inplace=True, axis=1)

    presets = {
        'xgb-tst5': {
            'model': utils.Xgb({
                'max_depth': 6,
                'eta': 0.15,
                'colsample_bytree': 0.9,
                'min_child_weight': 5,
                'subsample': 0.9,
                'early_stopping_rounds': 50
            }, n_iter=500)}
    }

    train_x = all_data.ix[all_data['click_day'] < 31, :].drop('label', axis=1)
    train_y = all_data.ix[all_data['click_day'] < 31, 'label']

    feature_names = list(all_data.columns)
    feature_names.remove('label')

    # high interaction by xgbfir

    print('xgbfi for feature interaction ...')
    presets['xgb-tst5']['model'].fit(X_train=train_x, y_train=train_y, feature_names=feature_names, name='xgb', xgbfir_tag=1, xgbfir_name='tst5')


if __name__ == '__main__':
    xgrfir()

