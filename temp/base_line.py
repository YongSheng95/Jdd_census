# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import datetime
import warnings
warnings.filterwarnings("ignore")

flow_train = pd.read_csv('data/flow_train.csv')
# tran_train = pd.read_csv('data/transition_train.csv')

# helper function
def _get_year(date_int):
    date = str(date_int)
    return int(date[0:4])


def _get_month(date_int):
    date = str(date_int)
    return int(date[4:6])


def _get_day(date_int):
    date = str(date_int)
    return int(date[6:8])


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))


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

# some feature
label = ['dwell','flow_in','flow_out']
feature = ['year','month','day']
result_attr = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']
result = pd.DataFrame(columns=result_attr)
df_temp = pd.DataFrame(columns=['pre_dwell', 'pre_flow_in', 'pre_flow_out'])

train_attr = ['date_dt','year','month','day', 'dwell','flow_in','flow_out']
train_feature = ['year','month','day', 'pre_dwell','pre_flow_in','pre_flow_out']

for ad in address:
    ad_split = ad.split(':')
    test_df['city_code'] = ad_split[0]
    test_df['district_code'] = ad_split[1]

    # add three features: pre_dwell, pre_flow_in, pre_flow_out
    train_data = flow_train[flow_train['address'] == ad][train_attr]
    train_data['index'] = [x for x in range(len(train_data))]
    df_temp['pre_dwell'] = train_data['dwell']
    df_temp['pre_flow_in'] = train_data['flow_in']
    df_temp['pre_flow_out'] = train_data['flow_out']
    df_temp['index'] = [x for x in range(1, 1 + len(df_temp))]
    train_data = pd.merge(train_data, df_temp, on='index')

    # train and validation data
    tr_data = train_data[train_data['date_dt'] < 20180214]
    val_data = train_data[train_data['date_dt'] >= 20180214]

    # training
    learners = []
    for y in label:
        # train_x = tr_data[train_feature]
        # train_y = tr_data[y]
        train_x = train_data[train_feature]
        train_y = train_data[y]
        val_x = val_data[train_feature]
        val_y = val_data[y]
        gbm = lgb.LGBMRegressor(boosting_type='gbdt',
            num_leaves=80,
            learning_rate=0.1,
            n_estimators=1000)
        gbm.fit(train_x, train_y,
                eval_set=[(val_x, val_y)],
                eval_metric='l2',
                early_stopping_rounds=5)
        y_pred = gbm.predict(val_x, num_iteration=gbm.best_iteration_)
        print('The rmsle of prediction is:', rmsle(val_y, y_pred))
        learners.append(gbm)

    # initialize 2018 03 01 data
    dwell_list = list(train_data[train_data['date_dt'] == 20180301]['dwell'])
    flow_in_list = list(train_data[train_data['date_dt'] == 20180301]['flow_in'])
    flow_out_list = list(train_data[train_data['date_dt'] == 20180301]['flow_out'])

    # predict
    for _, row in test_df.iterrows():
        x_feature = [row['year'], row['month'], row['day'], dwell_list[-1],
                     flow_in_list[-1], flow_out_list[-1]]
        test_x = np.array([x_feature])
        for i, learner in enumerate(learners):
            y_pre = learner.predict(test_x, num_iteration=learner.best_iteration_)
            if i == 0:
                dwell_list.append(y_pre[0])
            if i == 1:
                flow_in_list.append(y_pre[0])
            if i == 2:
                flow_out_list.append(y_pre[0])
        # print("predict:  ", dwell_list[-1],flow_in_list[-1], flow_out_list[-1])

    test_df['dwell'] = dwell_list[1:]
    test_df['flow_in'] = flow_in_list[1:]
    test_df['flow_out'] = flow_out_list[1:]
    result = pd.concat([result, test_df[result.columns]])


result['date_dt'] = result['date_dt'].astype(int)
result = result[['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']]
result.to_csv('prediction.csv', index=False, header=None)

print('ok')