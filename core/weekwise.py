from __future__ import division
import pandas as pd
import numpy as np
import json
import datetime
import calendar
import collections

df_sales = pd.read_csv('../data/Haiku_shirt_sales_analyzed.csv', parse_dates=['birth', 'order_date'],
                       infer_datetime_format=True, index_col = 'order_id')
df_sales.drop_duplicates(inplace=True)
df_sales.dropna(axis=0, subset=['total'], inplace=True)
print len(df_sales)
print df_sales.columns

df_sales.loc[df_sales['shirt_type'] == 'TShirt', 'shirt_type'] = 'T-Shirt'
print df_sales['shirt_type'].value_counts()

# Hoodies
df_sales = df_sales.loc[df_sales['shirt_type'] == 'Hoodie']

order_dates = np.array(df_sales['order_date'].values)
df_sales['order_date'] = pd.to_datetime(order_dates, errors='coerce') # , format='%d-%m-%Y %H:%M'
df_sales['order_year'], df_sales['order_month'], df_sales['order_day'], df_sales['order_hr'], \
df_sales['order_weekofyear'], df_sales['order_dayofweek'] = df_sales['order_date'].dt.year, \
                                                            df_sales['order_date'].dt.month, \
                                                            df_sales['order_date'].dt.day, \
                                                            df_sales['order_date'].dt.hour, \
                                                            df_sales['order_date'].dt.week, \
                                                            df_sales['order_date'].dt.dayofweek

df_sales = df_sales.sort('order_date')
# print '\n'.join([x.strftime('%d-%m-%Y %H:%M') for x in df_sales['order_date']])
grouped = df_sales.groupby(['order_year', 'order_weekofyear', 'order_dayofweek'])

data = collections.OrderedDict()
meanTemp = collections.OrderedDict()

counter = 0
for index, group in grouped:
    # print index
    year = index[0]
    week = index[1]
    weekday = index[2]
    # day_name = calendar.day_name[datetime.datetime(year=year, month=month, day=day).weekday()]
    key = str(year) + '-Wk' + str(week)
    sales = round(group['nb_tshirts'].sum(), 2)

    # print key, '=', sales
    if key in meanTemp:
        value = meanTemp[key]
        value[weekday] = sales
    else:
        value = [0.0] * 7
        value[weekday] = sales
        meanTemp[key] = value
print meanTemp
# info = collections.OrderedDict()
# for k in meanTemp:
#     vals = meanTemp[k]
#     mean = np.mean(vals)
#     text = ''
#     print k, mean
#
#     if mean > 30:
#         print 'high...'
#         class_name = 'hot'
#         text = 'Average Sales figures crossed 1k on ' + k
#     elif mean < 5:
#         print 'low...'
#         class_name = 'cold'
#         text = 'Average Sales figures pretty low on ' + k
#
#     if text != "":
#         if k in info:
#             value = {}
#             value['class'] = class_name
#             value['text'] = text
#         else:
#             value = {}
#             value['class'] = class_name
#             value['text'] = text
#             info[k] = value
#
# data["meanTemp"] = meanTemp
# data["info"] = info
#
# json_str = json.dumps(data)
# print json_str
#
# with open('data.json', 'w') as outfile:
#     json.dump(data, outfile)