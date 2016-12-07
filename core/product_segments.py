from __future__ import division
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import datetime

# plt.interactive(False)
df_sales = pd.read_csv('../data/Haiku_shirt_sales_analyzed.csv', parse_dates=['birth', 'order_date'],
                       infer_datetime_format=True, index_col = 'order_id')
print df_sales.columns
df_sales.loc[df_sales['shirt_type'] == 'TShirt', 'shirt_type'] = 'T-Shirt'
print df_sales['shirt_type'].value_counts()

df_sales.loc[df_sales['color'] == 'Bk', 'color'] = 'Black'
print df_sales['color'].value_counts()

# # clean data
# df_sales = df_sales.loc[df_sales['user_id'] != '0']
# df_sales['gender'] = df_sales['gender'].fillna('')
#
# # introducing expenditure segments
# expenditures = []
# ages = []
# df_users = df_sales.groupby(['user_id']).agg({
#     'total' : np.sum,
#     'nb_tshirts' : np.sum,
#     'age' : np.mean,
#     'gender' : lambda x: ' '.join(x.unique()).strip()
# })
# print df_users
#
# # introducing age buckets
# df_users['age_buckets'] = '60 and above'
# df_users.loc[df_users['age'] < 60, 'age_buckets'] = '40 to 60'
# df_users.loc[df_users['age'] < 40, 'age_buckets'] = '30 to 40'
# df_users.loc[df_users['age'] <= 30, 'age_buckets'] = '20 to 30'
# df_users.loc[df_users['age'] <= 20, 'age_buckets'] = '0 to 20'
#
# print df_users['age_buckets'].value_counts()
#
# # introducing age buckets
# df_users['expenditure_buckets'] = '600 and above'
# # df_users.loc[df_users['total'] < 1000, 'expenditure_buckets'] = '600 to 1000'
# df_users.loc[df_users['total'] < 600, 'expenditure_buckets'] = '300 to 600'
# df_users.loc[df_users['total'] <= 300, 'expenditure_buckets'] = '100 to 300'
# df_users.loc[df_users['total'] <= 100, 'expenditure_buckets'] = '0 to 100'
# # df_users.loc[df_users['total'] <= 50, 'expenditure_buckets'] = '0 to 50'
# print df_users['expenditure_buckets'].value_counts()
#
# # for index, group in df_groups:
# #     num_uniq = len(group['age'].unique())
# #     expenditure = group['total'].sum()
# #
# #     if num_uniq > 1:
# #         print index, expenditure, num_uniq
# #
# #     expenditures.append(expenditure)
# #     ages.append(num_uniq)
#
# # print 'Mean =', df_users['age'].mean()
# # print 'Max =', df_users['age'].max()
# # print 'Min =', df_users['age'].min()

# print df_sales
df_sales.to_csv('../data/Haiku_shirt_sales_products.csv', index=True, index_col = 'order_id')