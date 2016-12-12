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
df_sales['order_year'], df_sales['order_month'], df_sales['order_day'], df_sales['order_hr'] = \
    df_sales['order_date'].dt.year, df_sales['order_date'].dt.month, df_sales['order_date'].dt.day, \
    df_sales['order_date'].dt.hour

df_sales = df_sales.sort('order_date')
# print '\n'.join([x.strftime('%d-%m-%Y %H:%M') for x in df_sales['order_date']])
grouped = df_sales.groupby(['order_year', 'order_month', 'order_day', 'order_hr'])

data = collections.OrderedDict()
meanTemp = collections.OrderedDict()

counter = 0
for index, group in grouped:
    year = index[0]
    month = index[1]
    day = index[2]
    day_name = calendar.day_name[datetime.datetime(year=year, month=month, day=day).weekday()]
    month_name = datetime.datetime(year=year, month=month, day=day).strftime('%b')
    # key = str(year) + '-' + str(month) + '-' + str(day)
    key = month_name + ' ' + str(day) + ', ' + str(year) # + '(' + day_name[:3] + ')'
    sales = round(group['nb_tshirts'].sum(), 2)

    if day_name == 'Saturday' or day_name == 'Sunday':
        # print key, ':', day_name
        # print key, '=', sales
        if key in meanTemp:
            value = meanTemp[key]
            value[index[3]] = sales
        else:
            value = [0.0] * 24
            value[index[3]] = sales
            meanTemp[key] = value

    counter += 1
# print meanTemp.keys()
# print meanTemp.values()

# df_grouped = pd.DataFrame(data=meanTemp.values(), columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
#         '17', '18', '19', '20', '21', '22', '23'], index=meanTemp.keys())
# print df_grouped

info = collections.OrderedDict()
for k in meanTemp:
    vals = meanTemp[k]
    mean = np.mean(vals)
    text = ''
    print k, mean

    if mean > 5:
        print 'high...'
        class_name = 'hot'
        text = 'Average Sales figures crossed 1k on ' + k
    elif mean < 0.2:
        print 'low...'
        class_name = 'cold'
        text = 'Average Sales figures pretty low on ' + k

    if text != "":
        if k in info:
            value = {}
            value['class'] = class_name
            value['text'] = text
        else:
            value = {}
            value['class'] = class_name
            value['text'] = text
            info[k] = value

data["meanTemp"] = meanTemp
data["info"] = info

json_str = json.dumps(data)
print json_str

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)