# -*-coding:utf-8-*-

import os
# from my_tools import plot_utils

import joblib
import numpy as np
import pandas as pd
from script import utils

# from pyspark import SparkContext, sql

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


all_data['click_day'] = all_data['clickTime'].apply(lambda x: int(str(x)[:2]))
all_data['click_hour'] = all_data['clickTime'].apply(lambda x: int(str(x)[2:4]))
all_data['click_min'] = all_data['clickTime'].apply(lambda x: int(str(x)[4:]))


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
# plot_utils.plot_cate_bar(all_data.ix[(all_data['label'] != -1) & (all_data['diff_install_click'] > 0)], x='diff_install_click', y='label', estimator=np.mean)


# installed app category
print('Counting installed app category ...')
"""
user_installedapps = spark.read.csv(os.path.join('../dataset', 'user_installedapps.csv'), header=True)
app_categories = spark.read.csv(os.path.join('../dataset', 'app_categories.csv'), header=True)
category_count = user_installedapps.join(app_categories, 'appID', 'left_outer').groupBy(['userID', 'appCategory']).count().toPandas()
"""
category_count = joblib.load(os.path.join('../processed', 'userid_app_category_count'))

for column in category_count.columns:
    category_count[column] = category_count[column].astype(int)
category_count.rename(columns={'count': '0_category_count'}, inplace=True)

all_data = pd.merge(all_data, category_count, how='left', on=['userID', 'appCategory'])
all_data.ix[pd.isnull(all_data['0_category_count']), '0_category_count'] = 0


# user app actions
print('user app actions ...')
actions = pd.read_csv(os.path.join('../dataset', 'user_app_actions.csv'))
actions['install_day'] = actions['installTime'].apply(lambda x: utils.extract_day(x))

# check if user installed specific app-id in history
all_data = pd.merge(all_data, actions, how='left', on=['userID', 'appID'])
all_data['diff_install_click'] = all_data['install_day'] - all_data['click_day']

all_data.ix[all_data['diff_install_click'] < 0, 'diff_install_click'] = 1
all_data.ix[all_data['diff_install_click'] != 1, 'diff_install_click'] = 0


# check how much the same app category that the user clicked in a time-window
actions = pd.merge(actions, app_categories, how='left', on='appID')
grp_actions = actions.groupby(['userID', 'install_day', 'appCategory'], as_index=False)['installTime'].count().rename(columns={'installTime': 'install_counts'})


grp_app_install = pd.merge(all_data[['userID', 'appCategory', 'click_day']], grp_actions, how='left', on=['userID', 'appCategory'])
grp_app_install['diff_day'] = grp_app_install['click_day'] - grp_app_install['install_day']
grp_app_install = grp_app_install[pd.notnull(grp_app_install['diff_day'])]
grp_app_install = utils.calc_diff_day_category(grp_app_install)

all_data = pd.merge(all_data, grp_app_install, how='left', on=['userID', 'appCategory', 'click_day'])
all_data.drop(['installTime', 'install_day'], axis=1, inplace=True)

for day in range(1, 31):
    name = 'diff_' + str(day) + '_category'
    all_data.ix[pd.isnull(all_data[name]), name] = 0

print('After left-join day diff category, shape:', all_data.shape)


# encode with mean respond
print('encode with mean respond ...')

vn_list = ['creativeID', 'camgaignID',
           'advertiser_app_id', 'app_id_platform', 'appCategory',
           'site_position', 'hometown', 'residence', 'appID', 'positionID']


mean0 = all_data.ix[all_data['click_day'] < 31, 'label'].mean()
utils.calc_exptv(all_data, ['connectionType'], mean0)
utils.calc_exptv(all_data, vn_list)

print('After calc_exptv, shape:', all_data.shape)


print('Saving ...')
joblib.dump(all_data, os.path.join('../processed', 'all_data_p1'))
joblib.dump(instance_id, os.path.join('../processed', 'instance_id'))

