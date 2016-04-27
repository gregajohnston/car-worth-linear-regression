import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LinearRegression


df = pd.read_csv("car_data.csv", header=0)
# plt.xticks((14, 15, 16, 17, 18, 19, 20))
# plt.yticks((65, 70, 75, 80, 85, 90, 95, 100))


def calc_task_one_one():
    warnings.warn("deprecated", DeprecationWarning)
    model = LinearRegression()
    X = np.array(df['Mileage'].values).reshape(-1, 1)
    y = df['Price'].values
    model.fit(X, y)
    return model, X, y


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    calc_task_one_one()


def print_task_one_one(model):
    print("Line of best fit: ", end='')
    print('Y = {0:.2f}X'.format(model.coef_[0]), end='')
    print(' + {0:.2f}'.format(model.intercept_))


def draw_task_one_two(model, X, y):
    plt.scatter(X, y, color='b')
    plt.plot(X, model.predict(X), color='r', linewidth=2)
    plt.title('Price vs Mileage', fontsize=20)
    plt.xlabel('Mileage', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.xlim((-5000, 55000))
    plt.ylim((0, 75000))
    plt.show()


def print_task_one_three(model, X, y):
    print('Coefficient of determination: {0:.2f}'.format(model.score(X, y)))
