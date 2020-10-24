import math
import numpy as np


def regressao_linear(x, y):
    # number of observations/points 
    x = np.array(x)
    y = np.array(y)

    n = np.size(x)

    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients 
    m_0 = SS_xy / SS_xx
    b_1 = m_y - m_0 * m_x

    return (m_0, b_1)


def gaussiana(Y,m,X,b, var):
    gauss = 1 / (var * math.sqrt(2 * math.pi)) * math.exp(-1 / 2 * (((m*X+b - Y)** 2) / (var ** 2)))
    # gauss = E+var
    return gauss

def df1_mb(x, y, var, m, b):
    i = 0
    F = 0
    for x in X:
        # f = 1 / (var * math.sqrt(2 * math.pi)) * (2 * Y[i] * x - 2 * x * b - 2 * m * (x ** 2)) * math.exp(
        #     ((-Y[i] ** 2) + 2 * Y[i] * m * x - 2 * m * x * b + 2 * Y[i] * b - (m ** 2) * (x ** 2) - (b ** 2)) / (
        #                 2 * (var ** 2)))
        f = 1/(var * math.sqrt(2 * math.pi)) * (2*m*(x**2)+2*b*x-2*y[i]*x) * math.exp(-1*(((m*x+b-y[i])**2)/(2 * (var ** 2))))
        F = F + f
        i = i + 1
    return F


def df2_mb(x, y, var, m, b):
    i = 0
    F = 0
    for x in X:
        # f = 1 / (var * math.sqrt(2 * math.pi)) * (-2 * m * x + 2 * Y[i] - 2 * b) * math.exp(
        #     ((-Y[i] ** 2) + 2 * Y[i] * m * x - 2 * m * x * b + 2 * Y[i] * b - (m ** 2) * (x ** 2) - (b ** 2)) / (
        #                 2 * (var ** 2)))
        f = 1/(var * math.sqrt(2 * math.pi)) * (2*m*x+2*b-2*y[i]) * math.exp(-1*(((m*x+b-y[i])**2)/(2 * (var ** 2))))
        F = F + f
        i = i + 1
    return F

def gradient(Y, x, var,k):
    m = 1.5
    b = 2.5

    max_iters = 100000
    iters = 0

    df1 = lambda m, b: df1_mb(X, Y, var, m, b)
    df2 = lambda m, b: df2_mb(X, Y, var, m, b)

    while iters < max_iters:
        m_anterior = m
        b_anterior = b

        [m, b] = np.array([m_anterior, b_anterior]) - k * np.array([df1(m_anterior,b_anterior),df2(m_anterior,b_anterior)])

        iters = iters + 1
        print("Iteration", iters, "\nM value is", m, " and B is", b)

    print("The local maximum occurs at", m, b)
    return m, b

if __name__ == '__main__':
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Y = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    # Y = [5.007366366, 7.939129429, 9.384805457, 11.50568119, 13.54139611, 15.78721425, 17.65676338, 19.4641425,
    #      21.18284546, 23.89497607]
    # Y = [11.08214576, 15.20941695, 16.29593689, 15.21495305, 20.97790704, 21.87059378, 18.6563924, 22.68424192, 24.98079536, 32.78334686]

    E = np.subtract(X, Y)

    var = 0.5

    [m_reg, b_reg] = regressao_linear(X, Y)
    [m, b] = gradient(Y, X, var, k = 0.5)

    SomaG = 0
    i = 0
    for x in X:
        SomaG = SomaG + gaussiana(Y[i],m,x,b,var)
        i=i+1
    i