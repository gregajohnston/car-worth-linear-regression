import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LinearRegression


df = pd.read_csv("car_data.csv", header=0)
x_list = ['Mileage', 'Cylinder', 'Liter', 'Doors',
          'Cruise', 'Sound', 'Leather']


def calc_task_two_one():
    warnings.warn("deprecated", DeprecationWarning)
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    return model, X, y


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    calc_task_two_one()


def print_task_two_one(model):
    print("Line of best fit:\nY = ", end='')
    for idx in range(len(x_list)):
        print('{}*{} + '.format(format(model.coef_[idx], '.2f'),
                                x_list[idx]), end='')
    print('{}'.format(format(model.intercept_, '.2f')))


def print_task_two_two(model, X, y):
    print('Coefficient of determination: {0:.2f}'.format(model.score(X, y)))


def print_task_two_three():
    missing_liter()
    missing_sound()
    missing_mileage()
    missing_doors()
    missing_cylinder()
    missing_leather()
    missing_cruise()
    print("")
    missing_lit_sou()
    missing_lit_mil()
    missing_lit_doo()
    missing_lit_lea()
    missing_lit_cru()
    missing_lit_cyl()



def missing_lit_mil():
    x_list = ['Cylinder', 'Doors',
              'Cruise', 'Sound', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Liter/Mileage: {0:.2f}'.format(model.score(X, y)))


def missing_lit_cyl():
    x_list = ['Mileage', 'Doors',
              'Cruise', 'Sound', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Liter/Cylinder: {0:.2f}'.format(model.score(X, y)))


def missing_lit_doo():
    x_list = ['Mileage', 'Cylinder',
              'Cruise', 'Sound', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Liter/Doors: {0:.2f}'.format(model.score(X, y)))


def missing_lit_cru():
    x_list = ['Mileage', 'Cylinder',
              'Doors', 'Sound', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Liter/Cruise: {0:.2f}'.format(model.score(X, y)))


def missing_lit_sou():
    x_list = ['Mileage', 'Cylinder',
              'Doors', 'Cruise', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Liter/Sound: {0:.2f}'.format(model.score(X, y)))


def missing_lit_lea():
    x_list = ['Mileage', 'Cylinder',
              'Doors', 'Cruise', 'Sound']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Liter/Leather: {0:.2f}'.format(model.score(X, y)))


def missing_leather():
    x_list = ['Mileage', 'Cylinder', 'Liter',
              'Doors', 'Cruise', 'Sound']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Leather: {0:.2f}'.format(model.score(X, y)))


def missing_sound():
    x_list = ['Mileage', 'Cylinder', 'Liter',
              'Doors', 'Cruise', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Sound: {0:.2f}'.format(model.score(X, y)))


def missing_cruise():
    x_list = ['Mileage', 'Cylinder', 'Liter',
              'Doors', 'Sound', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Cruise: {0:.2f}'.format(model.score(X, y)))


def missing_doors():
    x_list = ['Mileage', 'Cylinder', 'Liter',
              'Cruise', 'Sound', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Doors: {0:.2f}'.format(model.score(X, y)))


def missing_liter():
    x_list = ['Mileage', 'Cylinder', 'Doors',
              'Cruise', 'Sound', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Liter: {0:.2f}'.format(model.score(X, y)))


def missing_cylinder():
    x_list = ['Mileage', 'Liter', 'Doors',
              'Cruise', 'Sound', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Cylinder: {0:.2f}'.format(model.score(X, y)))


def missing_mileage():
    x_list = ['Cylinder', 'Liter', 'Doors',
              'Cruise', 'Sound', 'Leather']
    model = LinearRegression()
    X = np.array(df[x_list].values)
    y = df['Price'].values
    model.fit(X, y)
    print('R**2 no Mileage: {0:.2f}'.format(model.score(X, y)))

    # plt.scatter(X, y, color='b')
    # plt.plot(X, model.predict(X), color='r', linewidth=2)
    # plt.title('Price vs Mileage', fontsize=20)
    # plt.xlabel('Mileage', fontsize=14)
    # plt.ylabel('Price', fontsize=14)
    # plt.xlim((-5000, 55000))
    # plt.ylim((0, 75000))
    # plt.show()
