from __future__ import division
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import datetime

def parse_date(start, end, td):
    if pd.isnull(start):
        return float('nan')

    resYear = td.days/364.0                   # get the number of years including the the numbers after the dot
    # resMonth = int((resYear - int(resYear))*364/30)  # get the number of months, by multiply the number after the dot by 364 and divide by 30.
    # resYear = int(resYear)
    # return str(resYear) + "Y" + str(resMonth) + "m"
    return resYear

# plt.interactive(False)
df_sales = pd.read_csv('../data/Haiku_shirt_sales.csv', parse_dates=['birth', 'order_date'],
                       infer_datetime_format=True, index_col = 'order_id')
df_sales.drop_duplicates(inplace=True)
df_sales.dropna(axis=0, subset=['total'], inplace=True)
print len(df_sales)
print df_sales.columns

order_dates = np.array(df_sales['order_date'].values)
df_sales['order_date'] = pd.to_datetime(order_dates, errors='coerce') # , format='%d-%m-%Y %H:%M'
df_sales['order_year'], df_sales['order_month'], df_sales['order_day'], df_sales['order_hr'], \
df_sales['order_weekofyear'], df_sales['order_dayofweek'] = df_sales['order_date'].dt.year, \
                                                            df_sales['order_date'].dt.month, \
                                                            df_sales['order_date'].dt.day, \
                                                            df_sales['order_date'].dt.hour, \
                                                            df_sales['order_date'].dt.week, \
                                                            df_sales['order_date'].dt.dayofweek
df_sales['min_order_date'] = df_sales['order_date'].min()

# age
df_sales['age'] = [parse_date(start, end, start - end)
                   for start, end in zip(df_sales['min_order_date'], df_sales['birth'])]

# # introducing age buckets
# df_sales['age_buckets'] = '60 and above'
# df_sales.loc[df_sales['age'] < 60, 'age_buckets'] = '40 to 60'
# df_sales.loc[df_sales['age'] < 40, 'age_buckets'] = '30 to 40'
# df_sales.loc[df_sales['age'] <= 30, 'age_buckets'] = '20 to 30'
# df_sales.loc[df_sales['age'] <= 20, 'age_buckets'] = '0 to 20'
#
# print df_sales['age_buckets'].value_counts()

# extract gender if present in shirt info
shirt_descs = np.array(df_sales['category'].values)
genders = []
colors = []
types = []
for desc in shirt_descs:

    # extract shirt gender
    if desc[-1].isupper():
        genders.append(desc[-1])
    else:
        genders.append('')

    # extract shirt color
    splits = desc.split()
    if len(splits) == 3:
        types.append(splits[1])
        colors.append(splits[0])
    else:
        types.append(' '.join(splits))
        colors.append('')

df_sales['shirt_type'] = types
df_sales['gender'] = genders
df_sales['color'] = colors
df_sales = df_sales.drop(['category', 'min_order_date'], 1)
print df_sales
df_sales.to_csv('../data/Haiku_shirt_sales_analyzed.csv', index=True, index_col = 'order_id')