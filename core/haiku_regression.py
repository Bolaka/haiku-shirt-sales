import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math

df_haiku = pd.read_csv('../data/Haiku_shirt_sales_analyzed.csv')
df_predictions = pd.DataFrame(columns=['Actuals', 'Predicteds'])

print df_haiku.columns

# target = nb_tshirts, regression problem
target = 'nb_tshirts'
features = ['departement', 'order_year', 'order_month', 'order_hr', 'order_dayofweek', 'is_birthday', '60 to 100',
            '40 to 60', '30 to 40', '20 to 30', '0 to 20', 'T-Shirt', 'Tennis Shirt', 'age']
# 'M', 'F', 'tshirt_price', 'Black', 'White'

features_all = ['departement', 'total', 'nb_tshirts', 'tshirt_price', 'order_year', 'order_month',
       'order_day', 'order_hr', 'order_weekofyear', 'order_dayofweek',
       'age', 'age_buckets', 'shirt_type', 'gender', 'color']

df_haiku = df_haiku.loc[df_haiku[target].notnull()]
df_haiku = df_haiku.loc[df_haiku['departement'].notnull()]

print 'length of data = ', str(len(df_haiku))

# # missing values
# df_haiku = df_haiku.fillna(-99.0,inplace=False)

# cleaning
df_haiku.loc[df_haiku['shirt_type'] == 'TShirt', 'shirt_type'] = 'T-Shirt'
df_haiku.loc[df_haiku['departement'] == '2A', 'departement'] = 2.1
df_haiku.loc[df_haiku['departement'] == '2B', 'departement'] = 2.5
# df_haiku.loc[df_haiku['age_buckets'] == 'outliers', 'age'] = -99.0
df_haiku.loc[df_haiku['color'] == 'Bk', 'color'] = 'Black'

print 'dummies for shirt_type:'
just_dummies = pd.get_dummies(df_haiku['shirt_type'], drop_first=True)
df_haiku = pd.concat([df_haiku, just_dummies], axis=1)

print 'dummies for age_buckets:'
just_dummies = pd.get_dummies(df_haiku['age_buckets'])
df_haiku = pd.concat([df_haiku, just_dummies], axis=1)

print 'dummies for gender:'
just_dummies = pd.get_dummies(df_haiku['gender'])
df_haiku = pd.concat([df_haiku, just_dummies], axis=1)

print 'dummies for color:'
just_dummies = pd.get_dummies(df_haiku['color'])
df_haiku = pd.concat([df_haiku, just_dummies], axis=1)

# spilt into 5-fold stratified training & test sets
sss = StratifiedKFold(df_haiku[target].values, 5, shuffle=True, random_state=786)
train_index, test_index = list(sss)[0]
print 'train size = ', len(train_index)
print 'test size = ', len(test_index)
y = np.sqrt(df_haiku[target].values)
# y = df_haiku[target].values)

counter = 0
important_words = []
rmses = []
r2s = []
actuals = []
preds = []
for train_index, test_index in sss:
    print
    print 'fold = ', str(counter)

    df_data = df_haiku.copy()[features]
    x_train, x_test = df_data.iloc[train_index].values, df_data.iloc[test_index].values
    y_train, y_test = y[train_index], y[test_index]

    # regressor = DecisionTreeRegressor(random_state=1002)
    regressor = RandomForestRegressor(n_estimators=100, random_state=1002)
    # regressor = GradientBoostingRegressor(random_state=1002)
    regressor.fit(x_train, y_train)
    indices = np.argsort(regressor.feature_importances_)
    # indices = indices[100:]
    important_words = list(reversed( [features[i] for i in indices] ))

    y_pred = regressor.predict(x_test)
    y_test = np.square(y_test)
    y_pred = np.square(y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    rmses.append(rmse)
    r2s.append(abs(r2))
    print 'fold rmse =', rmse
    print 'R2 = ', r2

    actuals.extend(y_test)
    preds.extend(y_pred)

    counter += 1

df_predictions['Actuals'] = actuals
df_predictions['Predicteds'] = preds

print
print 'R squared (mean) = ', round(np.mean(r2s), 2)
print 'root mean squared error (mean) = ', round(np.mean(rmses), 2)
print 'R squared = ', round(r2_score(actuals, preds), 2)
print 'root mean squared error = ', round(math.sqrt(mean_squared_error(actuals, preds)), 2)
# print df_haiku['nb_tshirts'].min(), df_haiku['nb_tshirts'].max()
df_predictions.to_csv('../data/haiku_regression_results.csv', index=False)

# persist the important words to disk
pd.DataFrame(important_words).to_csv('../data/important_features.csv')