import numpy as np

matrix = np.array([
    [9,2,3]
    ,
    [4,1,0]
    ,
    [1,2,9]
])

### Vectors
#* columns
r = np.column_stack([3,2])
s = np.column_stack([-1, 2])


# 1. Inner product ## Correct

def innerProduct(x,y):
    return np.sum(x*y)


print('r.s', (innerProduct(r, s)), 'This is commutative')
print('s.r', (innerProduct(s, r)), 'This is commutative')
print('s.s', (innerProduct(s, s)))
print('r.r', (innerProduct(r, r)))


# 2. Cosine and Dot Product ## Broken

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

print('Size', size(s1))
print('Size Squared', size(s1)**2)

##############################################