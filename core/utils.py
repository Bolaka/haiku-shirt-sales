print(__doc__)
import datetime
import calendar
import itertools
import matplotlib.pyplot as plt
import numpy as np
import collections
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plotHeatMap(df, annotate = False):
    if annotate:
        ax = sns.heatmap(df, annot=True, fmt="d", linewidths=.5, cmap="YlGnBu", cbar=False) # , square=True
    else:
        ax = sns.heatmap(df, linewidths=.5, cmap="YlGnBu", cbar=False) # , square=True

    # turn the axis label
    for item in ax.get_yticklabels():
        item.set_rotation(0)

    for item in ax.get_xticklabels():
        item.set_rotation(90)

    sns.plt.show()

def groupByDayHour(df):
    grouped = df.groupby(['order_year', 'order_month', 'order_day', 'order_hr'])

    data = collections.OrderedDict()
    meanTemp = collections.OrderedDict()
    for index, group in grouped:
        year = int(index[0])
        month = int(index[1])
        day = int(index[2])
        hr = int(index[3])
        day_name = calendar.day_name[datetime.datetime(year=year, month=month, day=day).weekday()]
        month_name = datetime.datetime(year=year, month=month, day=day).strftime('%b')
        # key = str(year) + '-' + str(month) + '-' + str(day)
        key = month_name + ' ' + str(day) + ', ' + str(year) # + '(' + day_name[:3] + ')'
        sales = int(group['nb_tshirts'].sum())

        if day_name == 'Saturday' or day_name == 'Sunday':
            # print key, ':', day_name
            # print key, '=', sales
            if key in meanTemp:
                value = meanTemp[key]
                value[hr] = sales
            else:
                value = [0.0] * 24
                value[hr] = sales
                meanTemp[key] = value


    df_grouped = pd.DataFrame(data=meanTemp.values(), columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
            '17', '18', '19', '20', '21', '22', '23'], index=meanTemp.keys())
    return df_grouped

def groupByIndexColumn(df, by_index, by_column, by_value):
    grouped = df.groupby([by_index, by_column])

    data_list = []
    for index, group in grouped:
        data = collections.OrderedDict()
        data[by_index] = index[0]
        data[by_column] = index[1]
        data[by_value] = int(group[by_value].sum())
        data_list.append(data)

    df_grouped = pd.DataFrame(data=data_list, columns=[by_index, by_column, by_value])
    df_grouped = df_grouped.pivot(by_index, by_column, by_value)
    return df_grouped