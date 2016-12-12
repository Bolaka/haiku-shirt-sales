from __future__ import division
import pandas as pd
import numpy as np
import json
import seaborn as sns
import utils

df_sales = pd.read_csv('../data/Haiku_shirt_sales_analyzed.csv', parse_dates=['birth', 'order_date'],
                       infer_datetime_format=True, index_col = 'order_id')
df_sales.drop_duplicates(inplace=True)
df_sales.dropna(axis=0, subset=['total'], inplace=True)
print len(df_sales)
print df_sales.columns

df_sales.loc[df_sales['shirt_type'] == 'TShirt', 'shirt_type'] = 'T-Shirt'
df_sales.loc[df_sales['color'] == 'Bk', 'color'] = 'Black'
print df_sales['shirt_type'].value_counts()
print df_sales['color'].value_counts()
print df_sales['is_birthday'].value_counts()

# Filter By Hoodies
# df_sales = df_sales.loc[df_sales['shirt_type'] == 'Hoodie']

# order_dates = np.array(df_sales['order_date'].values)
# df_sales['order_date'] = pd.to_datetime(order_dates, errors='coerce') # , format='%d-%m-%Y %H:%M'
# df_sales['order_year'], df_sales['order_month'], df_sales['order_day'], df_sales['order_hr'] = \
#     df_sales['order_date'].dt.year, df_sales['order_date'].dt.month, df_sales['order_date'].dt.day, \
#     df_sales['order_date'].dt.hour
#
# df_sales = df_sales.sort('order_date')
# # print '\n'.join([x.strftime('%d-%m-%Y %H:%M') for x in df_sales['order_date']])

# Time vs Time
df_grouped = utils.groupByDayHour(df_sales)
utils.plotHeatMap(df_grouped)


# User vs Time
df_grouped = utils.groupByIndexColumn(df_sales, 'age_buckets', 'order_month', 'nb_tshirts')
utils.plotHeatMap(df_grouped, True)

df_grouped = utils.groupByIndexColumn(df_sales, 'age_buckets', 'order_hr', 'nb_tshirts')
utils.plotHeatMap(df_grouped, True)

df_grouped = utils.groupByIndexColumn(df_sales, 'age_buckets', 'order_dayofweek', 'nb_tshirts')
utils.plotHeatMap(df_grouped, True)

df_grouped = utils.groupByIndexColumn(df_sales, 'age_buckets', 'is_birthday', 'nb_tshirts')
utils.plotHeatMap(df_grouped, True)

# User vs Prod
df_grouped = utils.groupByIndexColumn(df_sales, 'age_buckets', 'shirt_type', 'nb_tshirts')
utils.plotHeatMap(df_grouped, True)

df_grouped = utils.groupByIndexColumn(df_sales, 'age_buckets', 'departement', 'nb_tshirts')
utils.plotHeatMap(df_grouped)

df_grouped = utils.groupByIndexColumn(df_sales, 'age_buckets', 'color', 'nb_tshirts')
utils.plotHeatMap(df_grouped, True)

df_grouped = utils.groupByIndexColumn(df_sales, 'age_buckets', 'gender', 'nb_tshirts')
utils.plotHeatMap(df_grouped, True)

# Prod vs Prod
df_grouped = utils.groupByIndexColumn(df_sales, 'shirt_type', 'departement', 'nb_tshirts')
utils.plotHeatMap(df_grouped)

df_grouped = utils.groupByIndexColumn(df_sales, 'gender', 'departement', 'nb_tshirts')
utils.plotHeatMap(df_grouped)

df_grouped = utils.groupByIndexColumn(df_sales, 'color', 'departement', 'nb_tshirts')
utils.plotHeatMap(df_grouped)

df_grouped = utils.groupByIndexColumn(df_sales, 'color', 'gender', 'nb_tshirts')
utils.plotHeatMap(df_grouped, True)

# Time vs Prod
df_grouped = utils.groupByIndexColumn(df_sales, 'order_month', 'departement', 'nb_tshirts')
utils.plotHeatMap(df_grouped)

df_grouped = utils.groupByIndexColumn(df_sales, 'order_hr', 'departement', 'nb_tshirts')
utils.plotHeatMap(df_grouped)

df_grouped = utils.groupByIndexColumn(df_sales, 'is_birthday', 'departement', 'nb_tshirts')
utils.plotHeatMap(df_grouped)

df_grouped = utils.groupByIndexColumn(df_sales, 'is_birthday', 'shirt_type', 'nb_tshirts')
utils.plotHeatMap(df_grouped, True)