import numpy as np
import pandas as pd
import math

def sigma(x):
    return 1 / (1 + np.exp(-x))
def distribution(x):
    return np.exp(x) / np.sum(np.exp(x))
## Normal Gaussian Distribution

def binnomial(x):
    choice = input("built-in or custom: ")
    if choice == "built-in":
        return np.random.binomial(1, x)
    else:
        fx = 1 / sigma(matrix) * math.sqrt(1 / (2 * np.pi)) * np.exp(-1 / 2 * matrix ** 2)
## Normal Gaussian Distribution

def real_normal_gaussian(x, i):
    choice = input("built-in or custom: ")
    if choice == "built-in":
        return np.random.normal(0, 1)
    else:
        mu = 0
        sigma = 1
        fx = 1 / (sigma * math.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)
        return fx
## Predicted Normal Gaussian Distribution

def guess_normal_gaussian(z, i):
    choice = input("built-in or custom: ")
    if choice == "built-in":
        return np.random.normal(0, 1)
    else:
        mu = 0
        sigma = 1
        fx = 1 / (sigma * math.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((z - mu) / sigma) ** 2)
        return fx
## Gradient Descent

def wrongness(x, z, i):
    return (real_normal_gaussian(x, i) - guess_normal_gaussian(z, i)) ** 2

def gradient_descent(x, z, i):
    return wrongness(x, z, i) * guess_normal_gaussian(z, i) * (1 - guess_normal_gaussian(z, i))

x = {1, 4 ,6, 3}
z = {-5, 3, 2, 1}
i = 1
def test(x, z, i):
    for i in range(10):
        x = real_normal_gaussian(x, i)
        z = guess_normal_gaussian(z, i)
        print("x: ", x)
        print("z: ", z)
        print("wrongness: ", wrongness(x, z, i))
        print("gradient descent: ", gradient_descent(x, z, i))

test(x, z, i)