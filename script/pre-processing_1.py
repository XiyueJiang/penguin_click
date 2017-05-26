# -*-coding:utf-8-*-

import os
from my_tools import plot_utils

import joblib
import numpy as np
import pandas as pd
from script import utils

from pyspark import SparkContext, sql

"""
os.environ['PYSPARK_SUBMIT_ARGS']="--master local[*] pyspark-shell"
sc = SparkContext('local[*]')
spark = sql.SparkSession.builder.appName("").getOrCreate()
"""


train = pd.read_csv(os.path.join('../dataset', 'train.csv'))
test = pd.read_csv(os.path.join('../dataset', 'test.csv'))

instance_id = test['instanceID']
test.drop('instanceID', inplace=True, axis=1)

all_data = pd.concat([train, test])
print('All dataset shape:', all_data.shape)


ad = pd.read_csv(os.path.join('../dataset', 'ad.csv'))
all_data = pd.merge(all_data, ad, how='left', on='creativeID')
print('After left-join ad, shape:', all_data.shape)

app_categories = pd.read_csv(os.path.join('../dataset', 'app_categories.csv'))
all_data = pd.merge(all_data, app_categories, how='left', on='appID')
print('After left-join app_categories, shape:', all_data.shape)

position = pd.read_csv(os.path.join('../dataset', 'position.csv'))
all_data = pd.merge(all_data, position, how='left', on='positionID')
print('After left-join position, shape:', all_data.shape)

user = pd.read_csv(os.path.join('../dataset', 'user.csv'))
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

all_data['site_position'] = np.add(all_data.sitesetID.astype('str').values, all_data.positionType.astype('str').values)
all_data['app_id_platform'] = np.add(all_data.appID.astype('str').values, all_data.appPlatform.astype('str').values)
all_data['advertiser_app_id'] = np.add(all_data.advertiserID.astype('str').values, all_data.appID.astype('str').values)

all_data['site_position'] = all_data['site_position'].astype('category').values.codes
all_data['app_id_platform'] = all_data['app_id_platform'].astype('category').values.codes
all_data['advertiser_app_id'] = all_data['advertiser_app_id'].astype('category').values.codes


# using group-by to check encode corr variables
plot_utils.plot_cate_bar(all_data.ix[(all_data['label'] != -1) & (all_data['diff_install_click'] > 0)], x='diff_install_click', y='label', estimator=np.mean)


# installed app category
print('Counting installed app category ...')
"""
user_installedapps = spark.read.csv(os.path.join('../dataset', 'user_installedapps.csv'), header=True)
app_categories = spark.read.csv(os.path.join('../dataset', 'app_categories.csv'), header=True)
category_count = user_installedapps.join(app_categories, 'appID', 'left_outer').groupBy(['userID', 'appCategory']).count().toPandas()
"""
category_count = joblib.load(os.path.join('../processed', 'userid_app_category_count'))
category_count['userID'] = category_count['userID'].astype(int)
category_count['appCategory'] = category_count['appCategory'].astype(int)
category_count['count'] = category_count['count'].astype(int)

all_data = pd.merge(all_data, category_count, how='left', on=['userID', 'appCategory'])
all_data.ix[pd.isnull(all_data['count']), 'count'] = 0

# user app actions
print('user app actions ...')
actions = pd.read_csv(os.path.join('../dataset', 'user_app_actions.csv'))
all_data = pd.merge(all_data, actions, how='left', on=['userID', 'appID'])


all_data['install_day'] = all_data['installTime'].apply(lambda x: utils.extract_day(x))
all_data['click_day'] = all_data['click_day'].astype(int)
all_data['diff_install_click'] = all_data['install_day'] - all_data['click_day']
all_data.ix[all_data['diff_install_click'] < 0] = 1
all_data.ix[all_data['diff_install_click'] != 1] = 0


# encode with mean respond
print('encode with mean respond ...')
vn_list = ['connectionType', 'creativeID', 'camgaignID',
           'advertiser_app_id', 'app_id_platform', 'appCategory',
           'site_position', 'hometown', 'residence']


mean0 = all_data.ix[all_data['click_day'] < 31, 'label'].mean()
utils.calc_exptv(all_data, vn_list, mean0)


print('Saving ...')
joblib.dump(all_data, os.path.join('../processed', 'all_data_p1'))
joblib.dump(instance_id, os.path.join('../processed', 'instance_id'))

