from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
import xgboost as xgb
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.utils import check_random_state


os.sys.path.append("/mnt/trident/xiaolan/python/Contest/penguin_click/script")
os.chdir('/mnt/trident/xiaolan/python/Contest/penguin_click/script')


all_data = joblib.load('../processed/all_data_p2')
instance_id = joblib.load('../processed/instance_id')

cvrt_value = all_data['label']
day_values = all_data['click_day']

print("to encode categorical features using mean responses from earlier days -- multivariate")


vns = ['connectionType', 'creativeID', 'camgaignID','advertiser_app_id', 'app_id_platform', 'appCategory', 'site_position',  'appID', 'positionID']


card_summary = {}

for key in vns:
    print("Calculating %s" % key)
    grouped_table = all_data.groupby([key,'click_day'])['click_day'].apply(np.size).unstack()
    card_summary[key] = np.median(grouped_table.apply(np.mean, axis=1).reset_index())

print("to encode categorical features using mean responses from earlier days -- multivariate")


def mergeLeaveOneOut2(df, dfv, vn):
    _key_codes = df[vn].values.codes
    vn_yexp = 'exp2_' + vn
    grp1 = df[vn_yexp].groupby(_key_codes)
    _mean1 = grp1.aggregate(np.mean)
    _mean = _mean1[dfv[vn].values.codes].values
    _mean[np.isnan(_mean)] = _mean1.mean()
    return _mean


def calcLeaveOneOut2(df, vn, vn_y, cred_k, r_k, power, mean0=None, add_count=False):
    if mean0 is None:
        mean0 = df_yt[vn_y].mean() * np.ones(df.shape[0])
    _key_codes = df[vn].values.codes
    grp1 = df[vn_y].groupby(_key_codes)
    grp_mean = pd.Series(mean0).groupby(_key_codes)
    mean1 = grp_mean.aggregate(np.mean)
    sum1 = grp1.aggregate(np.sum)
    cnt1 = grp1.aggregate(np.size)

    # print sum1
    # print cnt1
    vn_sum = 'sum_' + vn
    vn_cnt = 'cnt_' + vn
    _sum = sum1[_key_codes].values
    _cnt = cnt1[_key_codes].values
    _mean = mean1[_key_codes].values
    # print _sum[:10]
    # print _cnt[:10]
    # print _mean[:10]
    # print _cnt[:10]
    _mean[np.isnan(_sum)] = mean0.mean()
    _cnt[np.isnan(_sum)] = 0
    _sum[np.isnan(_sum)] = 0
    # print _cnt[:10]
    _sum -= df[vn_y].values
    _cnt -= 1
    # print _cnt[:10]
    vn_yexp = 'exp2_' + vn
    #   df[vn_yexp] = (_sum + cred_k * mean0)/(_cnt + cred_k)
    diff = np.power((_sum + cred_k * _mean) / (_cnt + cred_k) / _mean, power)
    if vn_yexp in df.columns:
        df[vn_yexp] *= diff
    else:
        df[vn_yexp] = diff
    if r_k > 0:
        df[vn_yexp] *= np.exp((np.random.rand(np.sum(filter_train)) - .5) * r_k)
    if add_count:
        df[vn_cnt] = _cnt
    return diff


def logloss(pred, y, weight=None):
    if weight is None:
        weight = np.ones(y.size)

    pred = np.maximum(1e-7, np.minimum(1 - 1e-7, pred))
    return - np.sum(weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred))) / np.sum(weight)


def my_lift(order_by, p, y, w, n_rank, dual_axis=False, random_state=0, dither=1e-5, fig_size=None):
    gen = check_random_state(random_state)
    if w is None:
        w = np.ones(order_by.shape[0])
    if p is None:
        p = order_by
    ord_idx = np.argsort(order_by + dither * np.random.uniform(-1.0, 1.0, order_by.size))
    p2 = p[ord_idx]
    y2 = y[ord_idx]
    w2 = w[ord_idx]

    cumm_w = np.cumsum(w2)
    total_w = cumm_w[-1]
    r1 = np.minimum(n_rank, np.maximum(1,
                                       np.round(cumm_w * n_rank / total_w + .4999999)))

    df1 = pd.DataFrame({'r': r1, 'pw': p2 * w2, 'yw': y2 * w2, 'w': w2})
    grp1 = df1.groupby('r')

    sum_w = grp1['w'].aggregate(np.sum)
    avg_p = grp1['pw'].aggregate(np.sum) / sum_w
    avg_y = grp1['yw'].aggregate(np.sum) / sum_w

    xs = range(1, n_rank + 1)

    # fig, ax1 = plt.subplots()
    # if fig_size is None:
    #     fig.set_size_inches(20, 15)
    # else:
    #     fig.set_size_inches(fig_size)
    # ax1.plot(xs, avg_p, 'b--')
    # if dual_axis:
    #     ax2 = ax1.twinx()
    #     ax2.plot(xs, avg_y, 'r')
    # else:
    #     ax1.plot(xs, avg_y, 'r')

    print ("logloss: ", logloss(p, y, w))

    return gini_norm(order_by, y, w)


# def calc_exptv2(all_data, vns):

    df = all_data.ix[np.logical_and(all_data.click_day.values >= 17, all_data.click_day.values < 32), ['label', 'click_day'] + vns].copy()

    for vn in vns:
        df[vn] = df[vn].astype('category')
        print(vn)

    weight = {'connectionType': 1000, 'creativeID': 200, 'camgaignID': 200,'advertiser_app_id': 50,
             'app_id_platform': 20, 'appCategory': 200, 'site_position': 10,  'appID': 150, 'positionID': 100}

    exp2_dict = {}
    for vn in vns:
        exp2_dict[vn] = np.zeros(df.shape[0])

    days_npa = df.click_day.values

    for day_v in range(18, 32):
        df1 = df.ix[np.logical_and(df.click_day.values < day_v, df.click_day.values < 31), :].copy()
        df2 = df.ix[df.click_day.values == day_v, :]
        print("Validation day:", day_v, ", train data shape:", df1.shape, ", validation data shape:", df2.shape)
        pred_prev = df1.label.values.mean() * np.ones(df1.shape[0])

        for vn in vns:
            if 'exp2_' + vn in df1.columns:
                df1.drop('exp2_' + vn, inplace=True, axis=1)

        for i in range(5):
            for vn in vns:
                p1 = calcLeaveOneOut2(df1, vn, 'label', weight[vn], 0, 0.25, mean0=pred_prev)
                print(pred_prev, vn, p1)
                pred = pred_prev * p1
                print(day_v, i, vn, "change = ", ((pred - pred_prev) ** 2).mean())
                pred_prev = pred

            pred1 = df1.label.values.mean()
            for vn in vns:
                print("=" * 20, "merge", day_v, vn)
                diff1 = mergeLeaveOneOut2(df1, df2, vn)
                pred1 *= diff1
                exp2_dict[vn][days_npa == day_v] = diff1

            pred1 *= df1.label.values.mean() / pred1.mean()
            # print("logloss = ", logloss(pred1, df2.label.values))
            # print(my_lift(pred1, None, df2.label.values, None, 20, fig_size=(10, 5)))
            # plt.show()


def gini_norm(pred, y, weight=None):

    #equal weight by default
    if weight == None:
        weight = np.ones(y.size)

    #sort actual by prediction
    ord = np.argsort(pred)
    y2 = y[ord]
    w2 = weight[ord]

    #gini by pred
    cumm_y = np.cumsum(y2)
    total_y = cumm_y[-1]
    total_w = np.sum(w2)
    g1 = 1 - 2 * sum(cumm_y * w2) / (total_y * total_w)

    #sort actual by actual
    ord = np.argsort(y)
    y2 = y[ord]
    w2 = weight[ord]
    #gini by actual
    cumm_y = np.cumsum(y2)
    g0 = 1 - 2 * sum(cumm_y * w2) / (total_y * total_w)

    return g1/g0
