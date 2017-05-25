import os
from my_tools import plot_utils

import joblib



all_data = joblib.load(os.path.join('../processed', 'all_data_p1'))

id_columns = ['creativeID', 'positionID', 'adID', 'camgaignID',
              'advertiserID', 'appID', 'appPlatform', 'appCategory',
              'sitesetID', 'positionType']

# plot_utils.plot_correlation_map(all_data[id_columns])

all_data['site_position'] = all_data.add(all_data.sitesetID, )


