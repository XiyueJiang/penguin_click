# -*-coding:utf-8-*-

from collections import defaultdict
import numpy as np


def calc_exptv(df, vn_list, mean0):

    mean0 = mean0
    day_expt = defaultdict()
    df_copy0 = df.ix[:, ['click_day', 'label']].copy()
    day1 = 16
    cred_k = 11

    for vn in vn_list:

        print('exptv1', vn)
        df_copy0[vn] = df[vn]

        for day in range(17, 32):

            day_expt[day] = {}

            mask_prev = np.logical_and(df_copy0.click_day > day1, df_copy0.click_day <= day)
            mask_target = np.logical_and(df_copy0.click_day != day, df_copy0.click_day < 31)

            df_copy1 = df_copy0.ix[mask_prev, :].copy()

            df_prev = df_copy1.ix[mask_target, :].copy()

            vn_group = df_prev.groupby(vn)
            group_sum = vn_group['label'].aggregate(np.sum)
            group_cnt = vn_group['label'].aggregate(np.size)

            target = df_copy1.ix[~mask_target, vn].copy()
            _sum = group_sum[target].values
            _cnt = group_cnt[target].values

            _cnt[np.isnan(_sum)] = 0
            _sum[np.isnan(_sum)] = 0

            r = dict()
            exp = (_sum + cred_k * mean0) / (_cnt + cred_k)

            r['exp'] = exp
            r['cnt'] = _cnt

            day_expt[day][vn] = r

        df_copy0.drop(vn, inplace=True, axis=1)

    for vn in vn_list:
        vn_exp = 'exptv_' + vn
        df[vn_exp] = np.zeros(df.shape[0])

        for day in range(17, 32):
            df.loc[df.click_day.values == day, vn_exp] = day_expt[day][vn]['exp']

