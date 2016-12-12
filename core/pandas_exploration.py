# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Always a good practice to monitor time.
import time
start = time.strftime('%H:%M:%S')
print start

# Method to compute difference between 2 dates in years
def parse_date(start, end, td):
    if pd.isnull(start):
        return float('nan')

    resYear = td.days/364.0 # get the number of years including the the numbers after the dot
    return resYear

# load sales data of a fictional T-shirt making company called "Haiku T-Shirt".
df_sales = pd.read_csv('../data/Haiku_shirt_sales.csv', parse_dates=['birth', 'order_date'],
                       infer_datetime_format=True, index_col = 'order_id')

# Number fo Rows:
print 'No Of Rows = ', len(df_sales)

# Quick Exploration
# print the columns
print df_sales.columns

# first 10 rows, describe & frequency table
print df_sales.head(10)
print df_sales.describe()
print df_sales['category'].value_counts()

# print columns
# this is a datetime column
print df_sales['birth']

# extracting month, day from the birth datetime column
df_sales['birth_month'], df_sales['birth_day'] = df_sales['birth'].dt.month, df_sales['birth'].dt.day

# Feature Engineering
# let's convert order_date to datetime!
order_dates = np.array(df_sales['order_date'].values)
df_sales['order_date'] = pd.to_datetime(order_dates, errors='coerce')

# this is not a datetime column
print df_sales['order_date']

# extracting year from the datetime column
df_sales['order_year'] = df_sales['order_date'].dt.year
print df_sales['order_year']
# likewise, extracting month, day, hour, weekofyear, dayofweek
df_sales['order_month'], df_sales['order_day'], df_sales['order_hr'], \
df_sales['order_weekofyear'], df_sales['order_dayofweek'] = df_sales['order_date'].dt.month, \
                                                            df_sales['order_date'].dt.day, \
                                                            df_sales['order_date'].dt.hour, \
                                                            df_sales['order_date'].dt.week, \
                                                            df_sales['order_date'].dt.dayofweek

df_sales['is_birthday'] = 0
df_sales.loc[(df_sales['birth_month'] == df_sales['order_month'])
             & (df_sales['birth_day'] == df_sales['order_day']), 'is_birthday'] = 1
print df_sales['is_birthday'].value_counts()
df_sales['min_order_date'] = df_sales['order_date'].min()

# age
df_sales['age'] = [parse_date(start, end, start - end)
                   for start, end in zip(df_sales['min_order_date'], df_sales['birth'])]

# introducing age buckets
df_sales['age_buckets'] = 'outliers'
df_sales.loc[(df_sales['age'] >= 60) & (df_sales['age'] < 100), 'age_buckets'] = '60 to 100'
df_sales.loc[df_sales['age'] < 60, 'age_buckets'] = '40 to 60'
df_sales.loc[df_sales['age'] < 40, 'age_buckets'] = '30 to 40'
df_sales.loc[df_sales['age'] <= 30, 'age_buckets'] = '20 to 30'
df_sales.loc[(df_sales['age'] > 0) & (df_sales['age'] <= 20), 'age_buckets'] = '0 to 20'

print df_sales['age_buckets'].value_counts()

# extract gender if present in shirt info
print df_sales['category'].value_counts()

df_sales['category'] = df_sales['category'].fillna('')

shirt_descs = np.array(df_sales['category'].values)
genders = []
colors = []
types = []
for desc in shirt_descs:
    if desc == '':
        genders.append('')
        colors.append('')
        types.append('')
        continue

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
df_sales = df_sales.drop(['category', 'min_order_date', 'birth_month', 'birth_day'], 1)
print df_sales.columns

df_sales.to_csv('../data/Haiku_shirt_sales_analyzed.csv', index=True, index_col = 'order_id')


