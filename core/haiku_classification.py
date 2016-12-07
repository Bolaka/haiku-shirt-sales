import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sknn.mlp import Classifier, Layer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df_haiku = pd.read_csv('Haiku_shirt_sales_analyzed.csv')

# target = shirt-type, classification problem
target = 'shirt_type'
features = ['departement', 'age', 'order_year', 'order_month', 'order_day', 'order_hr',
            'order_dayofweek']
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

print 'cleaning color:'
print df_haiku['color'].value_counts()

print '\nfrequency table of target:'
print df_haiku[target].value_counts()

# spilt into 5-fold stratified training & test sets
sss = StratifiedKFold(df_haiku[target].values, 5, shuffle=True, random_state=786)
counter = 0
accuracies = []
for train_index, test_index in sss:
    # train_index, test_index = list(sss)[0]
    y = df_haiku[target].values
    y_train, y_test = y[train_index], y[test_index]
    df_data = df_haiku.copy()[features]
    # print
    # print 'fold = ', str(counter)
    # print 'train size = ', len(train_index)
    # print 'test size = ', len(test_index)
    x_train, x_test = df_data.iloc[train_index].values, df_data.iloc[test_index].values

    classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, random_state=1001)
    # classifier = LinearDiscriminantAnalysis()
    classifier.fit(x_train, y_train)
    classes = classifier.classes_

    y_pred = classifier.predict(x_test)
    accuracies.append(accuracy_score(y_test, y_pred))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    counter += 1

    break

print
print 'accuracy = ', round(np.mean(accuracies)*100, 2), '%'