# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import datetime
from jdd_census.model.lstm import LSTM


# lstm configuration
config = {'time_step': 20,'hidden_unit': 20, 'batch_size': 32,'input_size': 3,
          'output_size': 3,'learning_rate': 0.01,'epochs': 40}
# read data
flow_train = pd.read_csv('data/src/flow_train.csv')


def _get_year(date_int):
    date = str(date_int)
    return int(date[0:4])


def _get_month(date_int):
    date = str(date_int)
    return int(date[4:6])


def _get_day(date_int):
    date = str(date_int)
    return int(date[6:8])


# extract year, month, day
flow_train['year'] = flow_train['date_dt'].apply(_get_year)
flow_train['month'] = flow_train['date_dt'].apply(_get_month)
flow_train['day'] = flow_train['date_dt'].apply(_get_day)

# acquire unique address
flow_train['address'] = flow_train['city_code']+':'+flow_train['district_code']
address = list(set(flow_train['address']))
len_address = len(address)

# create test date
star = '20180301'
dates = []
for i in range(1,16):
    date_format = datetime.datetime.strptime(star,'%Y%m%d')
    fut_date = date_format + datetime.timedelta(days=i)
    dates.append(int(datetime.datetime.strftime(fut_date,'%Y%m%d')))
test_df = pd.DataFrame({'date_dt':dates})
test_df['year'] = test_df['date_dt'].apply(_get_year)
test_df['month'] = test_df['date_dt'].apply(_get_month)
test_df['day'] = test_df['date_dt'].apply(_get_day)
test_df['date_dt'] = test_df['date_dt'].astype(str)


feature = ['dwell', 'flow_in', 'flow_out']
results_attr = ['date_dt', 'city_code', 'district_code',
                'dwell', 'flow_in', 'flow_out']
results = pd.DataFrame(columns=results_attr)


def positive_number(x):
    if x < 0:
        return abs(x)*0.1
    return x


for i, ad in enumerate(address):
    ad_split = ad.split(':')
    test_df['city_code'] = ad_split[0]
    test_df['district_code'] = ad_split[1]

    data = np.array(flow_train[flow_train['address'] == ad][feature])
    normalize_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)

    # initialize first day data
    res_list = normalize_data[-config['time_step']:, :].tolist()
    lstm_model = LSTM(config, normalize_data, 'lstm' + str(i))
    for j in range(len(test_df)):
        test_x = res_list[-config['time_step']:]
        pred = lstm_model.predict(test_x)
        res_list.append(pred)
        # print('pred_{0}: {1}'.format(j, pred))

    result_ = np.array(res_list[-15:])
    dwell_list = (result_[:, 0]*std_data[0] + mean_data[0]).tolist()
    flow_in_list = (result_[:, 1]*std_data[1] + mean_data[1]).tolist()
    flow_out_list = (result_[:, 2]*std_data[2] + mean_data[2]).tolist()

    test_df['dwell'] = list(map(positive_number, dwell_list))
    test_df['flow_in'] = list(map(positive_number, flow_in_list))
    test_df['flow_out'] = list(map(positive_number, flow_out_list))
    results = pd.concat([results, test_df[results.columns]])

results['date_dt'] = results['date_dt'].astype(int)
results = results[
    ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']]
results.to_csv('data/res/prediction_lstm_1.csv', index=False, header=None)

