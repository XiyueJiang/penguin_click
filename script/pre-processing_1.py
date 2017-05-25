# -*-coding:utf-8-*-

import os
from my_tools import plot_utils

import joblib
import numpy as np
import pandas as pd

train = pd.read_csv(os.path.join('../dataset', 'train.csv'), dtype='str')
test = pd.read_csv(os.path.join('../dataset', 'test.csv'), dtype='str')

instance_id = test['instanceID']
test.drop('instanceID', inplace=True, axis=1)

all_data = pd.concat([train, test])
print('All dataset shape:', all_data.shape)


ad = pd.read_csv(os.path.join('../dataset', 'ad.csv'), dtype='str')
all_data = pd.merge(all_data, ad, how='left', on='creativeID')
print('After left-join ad, shape:', all_data.shape)

app_categories = pd.read_csv(os.path.join('../dataset', 'app_categories.csv'), dtype='str')
all_data = pd.merge(all_data, app_categories, how='left', on='appID')
print('After left-join app_categories, shape:', all_data.shape)

position = pd.read_csv(os.path.join('../dataset', 'position.csv'), dtype='str')
all_data = pd.merge(all_data, position, how='left', on='positionID')
print('After left-join position, shape:', all_data.shape)

user = pd.read_csv(os.path.join('../dataset', 'user.csv'), dtype='str')
all_data = pd.merge(all_data, user, how='left', on='userID')
print('After left-join user, shape:', all_data.shape)


all_data['click_day'] = all_data['clickTime'].apply(lambda x: str(x)[:2])
all_data['click_hour'] = all_data['clickTime'].apply(lambda x: str(x)[2:4])
all_data['click_min'] = all_data['clickTime'].apply(lambda x: str(x)[4:])


print('High corr id columns ...')
id_columns = ['creativeID', 'positionID', 'adID', 'camgaignID',
              'advertiserID', 'appID', 'appPlatform', 'appCategory',
              'sitesetID', 'positionType']

# plot_utils.plot_correlation_map(all_data[id_columns])

all_data['site_position'] = np.add(all_data.sitesetID.values, all_data.positionType.values)
all_data['app_id_platform'] = np.add(all_data.appID.values, all_data.appPlatform.values)
all_data['advertiser_app_id'] = np.add(all_data.advertiserID.values, all_data.appID.values)


print('Saving ...')
joblib.dump(all_data, os.path.join('../processed', 'all_data_p1'))
joblib.dump(instance_id, os.path.join('../processed', 'instance_id'))

