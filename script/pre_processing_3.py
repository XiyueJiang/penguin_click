# -*-coding:utf-8-*-

import os

import joblib

from script import utils


def main():

    column_pairs = [('connectionType', 'creativeID'),
                    ('connectionType', 'advertiser_app_id'),
                    ('connectionType', 'camgaignID'),
                    ('camgaignID', 'creativeID'),
                    ('advertiser_app_id', 'camgaignID'),
                    ('connectionType', 'app_id_platform'),
                    ('connectionType', 'positionType'),
                    ('connectionType', 'sitesetID'),
                    ('app_id_platform', 'creativeID')]

    all_data = joblib.load(os.path.join('../processed', 'all_data_p1'))
    utils.pair_interaction(all_data, column_pairs)

    print('After pair_interaction: ', all_data.shape)

    vn_list = ['_'.join(pair) for pair in column_pairs]

    mean0 = all_data.ix[all_data['click_day'] < 31, 'label'].mean()
    utils.calc_exptv(all_data, vn_list, mean0)

    print('After exptv pair_interaction: ', all_data.shape)

    print('Saving ...')
    joblib.dump(all_data, os.path.join('../processed', 'all_data_p2'))

if __name__ == '__main__':

    main()



