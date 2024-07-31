#!/usr/bin/env python
# coding: utf-8

1.
解马可夫矩阵最终稳态
2.
用马可夫矩阵和Eigenvector来建立网页追踪模型

#Eigenvalues and Eigenvectors
# - apply linear transformations, eigenvalues and eigenvectors in a webpage navigation model
### Packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg


# For the sake of the example consider there are only a small number of pages 5 x 5. 
# Transformation Matrix= P, 5 x 5 矩阵
# Define vector X, 初始状态 X0, Xn= P * Xn-1
例：
P = np.array([ 
    
    [0, 0.75, 0.35, 0.25, 0.85], 
    [0.15, 0, 0.35, 0.25, 0.05], 
    [0.15, 0.15, 0, 0.25, 0.05], 
    [0.15, 0.05, 0.05, 0, 0.05], 
    [0.55, 0.05, 0.25, 0.25, 0]  
]) 

初始状态假如是从第四个网页开始：
X0 = np.array([[0],[0],[0],[1],[0]])
那么第二个状态就是：
# Multiply matrix P and X_0 (matrix multiplication).
X1 = P @ X0

print(f'Sum of columns of P: {sum(P)}')
print(f'X1:\n{X1}')

# Applying the transformation m times you can find a vector Xm with the probabilities of the browser being at each of the pages after m steps of navigation.

迭代或变换20次
X = np.array([[0],[0],[0],[1],[0]])
m = 20

for t in range(m):
    X = P @ X
    
print(X)
# It is useful to predict the probabilities in Xm when m is large, and thus determine what pages a browser is more likely to visit after a long period of browsing the web. In other words, we want to know which pages ultimately get the most traffic. One way to do that is just apply the transformation many times, and with this small $5 \times 5$ example you can do that just fine. In real life problems, however, you'll have enormous matrices and doing so will be computationally expensive. Here is where eigenvalues and eigenvectors can help here significantly reducing the amount of calculations. Let's see how!
不断增加m，m=50, m=5000, m=50000, 最后出现 Xm+1 = Xm，系统进入稳态
这种稳态，表示为: 有 Vn+1 = Vn, 有P * Vn = 1 * Vn+1，则 Vn或Vn+1为Eigenvector with Eigenvalue = 1。 即稳态的Vn是转化矩阵P的Eigenvector以Eigenvalue为1。
###必须是Markov Matrix 才有这样的性质###

# Begin by finding eigenvalues and eigenvectors for the previously defined matrix P：
eigenvals, eigenvecs = np.linalg.eig(P)
print(f'Eigenvalues of P:\n{eigenvals}\n\nEigenvectors of P\n{eigenvecs}')

# In general, a square matrix whose entries are all nonnegative, and the sum of the elements for each column is equal to 1 is called a Markov matrix. 
# Markov matrices have a handy property - they always have an eigenvalue equal to 1. 
# You can easily verify that the matrix P you defined earlier is in fact a Markov matrix. 
# So, if  𝑚 is large enough, the equation  𝑋𝑚=𝑃𝑋𝑚−1
  can be rewritten as  𝑋𝑚=𝑃𝑋𝑚−1=1×𝑋𝑚
  This means that predicting probabilities at time  𝑚 , when  𝑚 is large you can simply just look for an eigenvector corresponding to the eigenvalue  1
# So, let's extract the eigenvector associated to the eigenvalue 1. 

X_inf = eigenvecs[:,0]

print(f"Eigenvector corresponding to the eigenvalue 1:\n{X_inf[:,np.newaxis]}")

Just to verify the results:
perform matrix multiplication  𝑃𝑋 (multiply matrix P and vector X_inf) to check that the result will be equal to the vector  𝑋 (X_inf).

def check_eigenvector(P, X_inf):
    X_check = P @ X_inf
    return X_check

X_check = check_eigenvector(P, X_inf)
print("Original eigenvector corresponding to the eigenvalue 1:\n" + str(X_inf))
print("Result of multiplication:" + str(X_check))

# Function np.isclose compares two NumPy arrays element by element, allowing for error tolerance (rtol parameter).
print("Check that PX=X element by element:" + str(np.isclose(X_inf, X_check, rtol=1e-10)))

This result gives the direction of the eigenvector, but as you can see the entries can't be interpreted as probabilities 
since you have negative values, and they don't add to 1. 
That's no problem. Remember that 
by convention np.eig returns eigenvectors with norm 1, but actually any vector on the same line is also an eigenvector to the eigenvalue 1, 
so you can simply scale the vector so that all entries are positive and add to one。This will give you the long-run probabilities of landing on a given web page。


X_inf = X_inf/sum(X_inf)
print(f"Long-run probabilities of being at each webpage:\n{X_inf[:,np.newaxis]}")

对最终稳态的翻译：
# This means that after navigating the web for a long time, 
the probability that the browser is at page 1 is 0.394, of being on page 2 is 0.134, on page 3 0.114, on page 4 0.085, and finally page 5 has a probability of 0.273.
# Looking at this result you can conclude that page 1 is the most likely for the browser to be at, while page 4 is the least probable one.
# If you compare the result of `X_inf` with the one you got after evolving the systems 20 times, they are the same up to the third decimal!
# Here is a fun fact: this type of a model was the foundation of the PageRank algorithm, which is the basis of Google's very successful search engine.

这是最重要且流行的网页追踪算法的基础！！！July/2024
