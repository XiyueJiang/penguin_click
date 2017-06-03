from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import sys
import pandas as pd
import os

os.sys.path.append("/mnt/trident/xiaolan/python/Contest/penguin_click/script")
os.chdir('/mnt/trident/xiaolan/python/Contest/penguin_click/script')

import utils

clf = RandomForestClassifier(n_estimators=40, max_depth = 10, min_samples_split=20, min_samples_leaf=5, random_state=0, criterion='entropy',
                             max_features=10, verbose=1, n_jobs=-1, bootstrap=False)

all_data = joblib.load('../processed/all_data_p2')
instance_id = joblib.load('../processed/instance_id')

cvrt_value = all_data['label']
day_values = all_data['click_day']

day_test = 30

drop_list = set(["diff_"+str(i)+"_category" for i in range(31)])
rf_feature = list(set(all_data.columns) - set(["clickTime", "conversionTime", "label", "click_min", "click_day","telecomsOperator","exptv_residence","marriageStatus","appPlatform","userID"]) - drop_list)

xv = all_data.ix[day_values == day_test, rf_feature]
yv = all_data.ix[day_values == day_test, 'label']
xt = all_data.ix[day_values < day_test, rf_feature]
yt = all_data.ix[day_values < day_test, 'label']
nn = all_data[day_values < day_test].shape[0]


def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)
    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)


ctr = 0
predv = 0

for ind in range(4):
    clf.random_state = ind
    np.random.seed(ind)
    r1 = np.random.uniform(0, 1, nn)
    filter1 = r1 < .5

    xt1 = xt.ix[filter1, rf_feature]
    yt1 = yt[filter1]

    rf1 = clf.fit(xt1, yt1)
    y_val_hat = rf1.predict_proba(xv.ix[:, rf_feature])[:, 1]

    y_train_hat = rf1.predict_proba(xt.ix[:, rf_feature])[:, 1]

    predv += y_val_hat
    ctr += 1

    val_ll = logloss(predv/ctr, yv)
    train_ll = logloss(y_train_hat, yt)

    print("iter", ind, ", validation logloss = ", val_ll)
    print("iter", ind, ", train logloss = ", train_ll)

    sys.stdout.flush()

    # rf1_imp = pd.DataFrame({'feature': rf_feature, 'impt': clf.feature_importances_})
    #print(rf1_imp.sort('impt', ascending=False).feature.head(50))

