import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sknn.mlp import Classifier, Layer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math

df_haiku = pd.read_csv('Haiku_shirt_sales_analyzed.csv')
df_predictions = pd.DataFrame(columns=['Actuals', 'Predicteds'])

# target = nb_tshirts, regression problem
target = 'nb_tshirts'
features = ['departement', 'order_year', 'order_month', 'order_day', 'order_hr',
            'order_dayofweek', 'age', 'T-Shirt', 'Tennis Shirt', 'tshirt_price']
features_all = ['departement', 'total', 'nb_tshirts', 'tshirt_price', 'order_year', 'order_month',
       'order_day', 'order_hr', 'order_weekofyear', 'order_dayofweek',
       'age', 'age_buckets', 'shirt_type', 'gender', 'color']

df_haiku = df_haiku.loc[df_haiku[target].notnull()]
df_haiku = df_haiku.loc[df_haiku['departement'].notnull()]

print 'length of data = ', str(len(df_haiku))

# missing values
df_haiku = df_haiku.fillna(-99.0,inplace=False)

# cleaning
df_haiku.loc[df_haiku['gender'] == 'F', 'gender'] = 0
df_haiku.loc[df_haiku['gender'] == 'M', 'gender'] = 1
df_haiku.loc[df_haiku['shirt_type'] == 'TShirt', 'shirt_type'] = 'T-Shirt'
df_haiku.loc[df_haiku['departement'] == '2A', 'departement'] = 2.1
df_haiku.loc[df_haiku['departement'] == '2B', 'departement'] = 2.2
df_haiku.loc[df_haiku['age_buckets'] == 'outliers', 'age'] = -99.0
df_haiku.loc[df_haiku['color'] == 'Bk', 'color'] = 'Black'
df_haiku.loc[df_haiku['color'] == 'Black', 'color'] = 1
df_haiku.loc[df_haiku['color'] == 'White', 'color'] = 0

print 'dummies for shirt_type:'
just_dummies = pd.get_dummies(df_haiku['shirt_type'], drop_first=True)
df_haiku = pd.concat([df_haiku, just_dummies], axis=1)

# spilt into 5-fold stratified training & test sets
sss = StratifiedKFold(df_haiku[target].values, 5, shuffle=True, random_state=786)
counter = 0
rmses = []
actuals = []
preds = []
for train_index, test_index in sss:
    # train_index, test_index = list(sss)[0]
    y = df_haiku[target].values
    y_train, y_test = y[train_index], y[test_index]
    df_data = df_haiku.copy()[features]
    print
    print 'fold = ', str(counter)
    # print 'train size = ', len(train_index)
    # print 'test size = ', len(test_index)
    x_train, x_test = df_data.iloc[train_index].values, df_data.iloc[test_index].values

    # classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, random_state=1001)
    # classifier = LinearDiscriminantAnalysis()
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    rmses.append(rmse)
    print 'fold rmse =', rmse
    print 'R2 = ', r2_score(y_test, y_pred)

    actuals.extend(y_test)
    preds.extend(y_pred)

    counter += 1

df_predictions['Actuals'] = actuals
df_predictions['Predicteds'] = preds

print
print 'root mean squared error = ', round(np.mean(rmses), 2)
print df_haiku['nb_tshirts'].min(), df_haiku['nb_tshirts'].max()
df_predictions.to_csv('haiku_regression_results.csv', index=False)