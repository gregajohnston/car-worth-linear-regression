import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LinearRegression
# df.drop('column_name', axis=1, inplace=True)

df = pd.read_csv("car_data.csv", header=0)
df.drop('Trim', axis=1, inplace=True)

df_make = pd.get_dummies(df['Make'], prefix='Make')
df = pd.concat([df, df_make], axis=1)
df.drop('Make', axis=1, inplace=True)

df_model = pd.get_dummies(df['Model'], prefix='Model')
df = pd.concat([df, df_model], axis=1)
df.drop('Model', axis=1, inplace=True)

df_type = pd.get_dummies(df['Type'], prefix='Type')
df = pd.concat([df, df_type], axis=1)
df.drop('Type', axis=1, inplace=True)


def print_task_three_one():
    print(list(df.columns))

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


def missing_lit():
    df.drop('Liter', axis=1, inplace=True)
    model = LinearRegression()
    X = np.array(df.values)
    y = s.values
    model.fit(X, y)
    print('R^2 no Liter: {0:.2f}'.format(model.score(X, y)))


def missing_lit_sou():
    df.drop('Sound', axis=1, inplace=True)
    model = LinearRegression()
    X = np.array(df.values)
    y = s.values
    model.fit(X, y)
    print('R^2 no Liter/Sound: {0:.2f}'.format(model.score(X, y)))


def missing_lit_sou_doo():
    df.drop('Doors', axis=1, inplace=True)
    model = LinearRegression()
    X = np.array(df.values)
    y = s.values
    model.fit(X, y)
    print('R^2 no Liter/Sound/Doors: {0:.2f}'.format(model.score(X, y)))


def missing_lit_sou_doo_mil():
    df.drop('Mileage', axis=1, inplace=True)
    model = LinearRegression()
    X = np.array(df.values)
    y = s.values
    model.fit(X, y)
    print('R^2 no Liter/Sound/Doors/Mileage: {0:.2f}'
          .format(model.score(X, y)))


def missing_lit_sou_doo_mil_lea():
    df.drop('Leather', axis=1, inplace=True)
    model = LinearRegression()
    X = np.array(df.values)
    y = s.values
    model.fit(X, y)
    print('R^2 no Liter/Sound/Doors/Mileage/Leather: {0:.2f}'
          .format(model.score(X, y)))
