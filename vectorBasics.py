%%timeit
import numpy as np
import pandas as pd
import math
import tensorflow as tf

np.random.seed(42)
matrix = np.random.rand(144, 144)
### Vectors
#* columns
r = np.column_stack([10 ,2 ,4, -1])
s = np.column_stack([0, -9, 3, 4])


# 1. Inner product ## Correct

def innerProduct(x,y):
    return np.sum(x*y)


print('r.s', (innerProduct(r, s)), 'This is commutative')
print('s.r', (innerProduct(s, r)), 'This is commutative')
print('s.s', (innerProduct(s, s)), 'This is distributive')
print('r.r', (innerProduct(r, r)), 'This is distributive')

def products(x, y):
    rsubs = innerProduct(r, s)
    ssubr = innerProduct(s, r)
    ssubs = innerProduct(s, s)
    rsubr = innerProduct(r, r)
    return rsubs, ssubr, ssubs, rsubr
print('products', products(r,s))


# 2. Cosine and Dot Product ## Broken
# Cosine  == c^2 = a^2 + b^2 - 2ab cos(theta)

def cosine(x, y):
    x = x
    y = y
    return x - 2*innerProduct(x,y) + abs(y)^2

print('Dot Product', cosine(r,s))

# 3. Projection or Orthogonal Projection  ##Broken

def projection(x, y):
    return innerProduct(x,y)/innerProduct(x,x) * x

print('Projection', projection(r,s))

## Size of vector ###################### Correct

s1 = np.column_stack([1,3,4,2])

# def size(x):
#     x1 = x**2
#     sum = np.sum(x1)
#     return np.sqrt(sum)

# print('size', size(s1))
# print('sqrt', size(s1)**2)

def size(x):
    return np.sqrt(innerProduct(x,x))

def assignedSizes(s,r):
    s1 = size(s) ## |s|
    r1 = size(r) ## |r|
    left = ((s1-r1)**2)
    right = ((r1**2) + (s1**2) - (2*r1*s1))
    rsubs = innerProduct(r, s)
    cosTheta = rsubs/(r1*s1)
    theta_radians = math.acos(cosTheta)
    theta = math.degrees(theta_radians)
    return cosTheta , theta

## Should I even have a Cosine function?? cosine is constant.

print('Finding Cosine and Angle of 2 Vectors')
print('Cosine of r on s & Angle of Vectos is: ', assignedSizes(s, r))
# print('Sizes(moduli) of s, r:', assignedSizes(s, r))
# print('Size Squared', size(r)**2)

# print('Size', size(s))
# print('Size Squared', size(s)**2)

##############################################


print(' ')

