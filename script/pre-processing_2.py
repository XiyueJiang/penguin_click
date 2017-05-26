# -*-coding:utf-8-*-

import os

import joblib
import utils

all_data = joblib.load(os.path.join('../processed', 'all_data_p1'))

# encode with mean respond
vn_list = ['connectionType', 'creativeID', 'camgaignID',
           'advertiser_app_id', 'app_id_platform', 'appCategory',
           'site_position', 'hometown', 'residence']

all_data['click_day'] = all_data['click_day'].astype(int)
all_data['label'] = all_data['label'].astype(int)

mean0 = all_data.ix[all_data['click_day'] < 31, 'label'].mean()
utils.calc_exptv(all_data, vn_list, mean0)

print(all_data.head())
