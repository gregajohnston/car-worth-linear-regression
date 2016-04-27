import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LinearRegression
# df.drop('column_name', axis=1, inplace=True)

df = pd.read_csv("car_data.csv", header=0)
df.drop('Trim', axis=1, inplace=True)

make_dummies = pd.get_dummies(df['Make'], prefix='Make')
df = df.join(make_dummies.ix[:, 'Make_2':])
df.drop('Make', axis=1, inplace=True)

make_dummies = pd.get_dummies(df['Model'], prefix='Model')
df = df.join(make_dummies.ix[:, 'Model_2':])
df.drop('Model', axis=1, inplace=True)

make_dummies = pd.get_dummies(df['Type'], prefix='Type')
df = df.join(make_dummies.ix[:, 'Type_2':])
df.drop('Type', axis=1, inplace=True)


def print_task_three_one():
    print(df.columns)

s = df['Price']
df.drop('Price', axis=1, inplace=True)


def calc_task_three_two():
    warnings.warn("deprecated", DeprecationWarning)
    model = LinearRegression()
    X = np.array(df.values)
    y = s.values
    model.fit(X, y)
    return model, X, y


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    calc_task_three_two()


def print_task_three_two_a(model):
    print("Line of best fit:\nY = ", end='')
    for idx in range(len(df.columns)):
        print('{}*{} + '.format(format(model.coef_[idx], '.2f'),
                                df.columns[idx]), end='')
    print('{}'.format(format(model.intercept_, '.2f')))


def print_task_three_two_b(model, X, y):
    print('\nCoefficient of determination: {0:.2f}'.format(model.score(X, y)))
