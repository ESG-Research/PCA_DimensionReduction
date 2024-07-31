

###Application of Eigenvalues and Eigenvectors: Principal Component Analysis (PCA)
# Packages：
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

算法，流程概述：
#To apply PCA on any dataset 
#you will begin by defining the covariance matrix. 
#After that you will compute the eigenvalues and eigenvectors of this covariance matrix. 
#Each of these eigenvectors will be a “principal component”. 
#To perform the dimensionality reduction, you will take the k principal components associated to the k biggest eigenvalues, 
#and transform the original data by projecting it onto the direction of these principal components (eigenvectors).


###第一步：Load the data导入和【逆可视化】图像进入一个 Dataset Array，便于进行后续操作（PCA）。

#Population是 ‘Cat and dog face’ dataset from Kaggle. 我们处理的Dataset是其中的 Cat Face。
# Begin by loading the images and transforming them to black and white using `load_images` function from utils. 
imgs = utils.load_images('./data/')

# imgs 就是图像的dataset，每个观测点就是一张图像(猫头)，这里每个图像就是一个矩阵Matrix，矩阵的每个element就是一个pixel像素点。

#如下代码操作，可以查看这里共有几张图象，每张图片是 n x n的矩阵 (什么样的矩阵形状Shape)。

height, width = imgs[0].shape
print(f'\nYour dataset has {len(imgs)} images of size {height}x{width} pixels\n')
# 结果：Your dataset has 55 images of size 64x64 pixels
#共有55张图片在这个dataset，每张图片都是 Matrix 64 x 64，即我们通常说的 “64像素”。

# Go ahead and plot one image to see what they look like. You can use the colormap 'gray' to plot in black and white. 
#改变其中imgs[]中的参数来查看dataset中的不同图像（矩阵）。
plt.imshow(imgs[0], cmap='gray')


# When working with images, you can consider each pixel as a variable. 每个图片是 64 x 64Matrix 意味着总共64个变量。
# 我们现在要操作这个含有64个变量Xi，和55个观测点的 Dataset。首先要把image变成一般的统计数据形式。也可以称为【逆可视化】。
# Having each image in matrix form is good for visualizing the image, but not so much for operating on each variable. 
# In order to apply PCA for dimensionality reduction 
# You will need to flatten each image into a single row vector. You can do this using the `reshape` function from NumPy. 操作代码如下：

imgs_flatten = np.array([im.reshape(-1) for im in imgs])

print(f'imgs_flatten shape: {imgs_flatten.shape}')

# The resulting array will have 55 rows, one for each image, and 64x64=4096 columns.
#现在变成了一个 55 x 4096的大Array，作为我们的数据集 Dataset Array，方便我们施加操作，如PCA变换。



###第二步： 找到协方差矩阵 Get the covariance matrix：
# Now that you have the images in the correct shape you are ready to apply PCA on the flattened dataset. 
# If you consider each pixel (column) as a variable, and each image (rows) as an obervation you will have 55 observations of 4096 variable
#数据集： 每个像素点（column）作为一个变量Xi，每张图片（row）作为一个观测点Oservation，该数据Array共有55个观测点，4096个变量。

现在第一步是：【中心化】Xi - meanXi 找到 Dataset Array的中心化矩阵Centered Matrix。
# In order to get the covariance matrix you first need to center the data by subtracting the mean for each variable (column). 
利用下面三大公式Functions做到这一点：
np.mean: use this function to compute the mean of each variable, just remember to pass the correct axis argument.
np.repeat: This will allow for you to repeat the values of each  𝜇𝑖
np.reshape: Use this function to reshape the repeated values into a matrix of shape the same size as your input data. To get the correct matrix after the reshape, remember to use the parameter order='F'.

def center_data(Y):
    """
    Center your original data
    Args:
         Y (ndarray): input data. Shape (n_observations x n_pixels)
    Outputs:
        X (ndarray): centered data
    """
    mean_vector = np.mean(Y, axis=0)
    mean_matrix = np.repeat(mean_vector,Y.shape[0],axis=0)
    # use np.reshape to reshape into a matrix with the same size as Y. Remember to use order='F'
    mean_matrix = np.reshape(mean_matrix,Y.shape,order='F')
    
    X = Y - mean_matrix
    return X
    
这里的X就是中心化矩阵，Dataset Array的Centered Matrix，它应该和Dataset Array， Y 有一样的形状Shpae，正如Reshape Function做到的。
#注意 axis=0或axis=1 对行，对列的参数设置。以及 X.shape[0 或 1]或者X.shape来得到矩阵X的行数，列数，和行列数的代数值。

# Go ahead and apply the `center_data` function to your data in `imgs_flatten`. 
# You can also print the image again and check that the face of the cat still looks the same. 
# This is because the color scale is not fixed, but rather relative to the values of the pixels. 

X = center_data(imgs_flatten)
plt.imshow(X[0].reshape(64,64), cmap='gray')

第二步：用【中心化】矩阵找到【Covariance Matrix】协方差矩阵
公式：C = （1/n-1）* (X转置 * X)   X是centered matrix
点乘  np.dot(X1 ,X2 )
转置  np.transpose(X)
# Now that you have your centered data, X, you can go ahead and find the covariance matrix 
# The covariance matrix can be found by appliying the dot product between np.transpose(X) and X, and divide by the number of observations minus 1.
# To perform the dot product you can simply use the function np.dot()

def get_cov_matrix(X):
    """ Calculate covariance matrix from centered data X
    Args:
        X (np.ndarray): centered data matrix
    Outputs:
        cov_matrix (np.ndarray): covariance matrix
    """
    cov_matrix = np.dot(np.transpose(X),  X)
    cov_matrix = cov_matrix/(X.shape[0]-1)
    
    return cov_matrix   

cov_matrix = get_cov_matrix(X)
# Check the dimensions of the covariance matrix, it should be a square matrix with 4096 rows and columns. 
print(f'Covariance matrix shape: {cov_matrix.shape}') 
结果：4096 x 4096


###第三步：计算协方差矩阵C的Eigenvalues和Eigenvectors。Compute the eigenvalues and eigenvectors：

# Now you are all set to compute the eigenvalues and eigenvectors of the covariance matrix.
# Due to performance constaints, you will not be using 公式： np.linalg.eig 
# But rather the very similar function 公式： scipy.sparse.linalg.eigsh
# This function allows you to compute fewer number of eigenvalue-eigenvector pairs. 运算经济，适用于小型计算机

# it can be shown that at most 55 eigenvalues of C will be different from zero, which is the smallest dimension of the data matrix X. 
# Thus, for computational efficiency, you will only be computing the first biggest 55 Eigenvalues
# and their corresponding Eigenvectors
# Feel free to try changing the 参数 “k” parameter in 公式 “scipy.sparse.linalg.eigsh” to something slightly bigger, 
# to verify that all the new eigenvalues are zero. Try to keep it below 80, otherwise it will take too long to compute. 
# The outputs of this scipy function are exactly the same as the ones from `np.linalg.eig`, except eigenvalues are ordered in decreasing order, so if you want to check out the largest eigenvalue you need to look into the last position of the vector. 

scipy.random.seed(7)
eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(cov_matrix, k=55)
print(f'Ten largest eigenvalues: \n{eigenvals[-10:]}')

保持每次运算EigenVector方向的一致性：用scipy.random.seed(7)
# The random seed is fixed in the code above to help ensure the same eigenvectors are calculated each time. 
# This is because for each eigenvector, there are actually two possible outcomes with norm 1. They fall on the same line but point in opposite directions. 
# Both possibilities are correct, but by fixing the seed you guarantee you will always get the same result. 
# In order to get a consistent result with 公式：“np.linalg.eig”, you will invert the order of `eigenvals` and `eigenvecs`
# so they are both ordered from largest to smallest eigenvalue.

eigenvals = eigenvals[::-1]
eigenvecs = eigenvecs[:,::-1]

print(f'Ten largest eigenvalues: \n{eigenvals[:10]}')


# Each of the eigenvectors you found will represent one principal component. 
# The eigenvector associated with the largest eigenvalue will be the first principal component, 
# the eigenvector associated with the second largest eigenvalue will be the second principal component, and so on. 
# It is pretty interesting to see that each principal component usually extracts some relevant features, or patterns from each image. 
# In the next cell you will be visualizing the first sixteen components to see that：

fig, ax = plt.subplots(4,4, figsize=(20,20))
for n in range(4):
    for k in range(4):
        ax[n,k].imshow(eigenvecs[:,n*4+k].reshape(height,width), cmap='gray')
        ax[n,k].set_title(f'component number {n*4+k+1}')


# What can you say about each of the principal components? 

# <a name='2.4'></a>
# ### 2.4 Transform the centered data with PCA
# 
# Now that you have the first 55 eigenvalue-eivenvector pairs, you can transform your data to reduce the dimensions. Remember that your data originally consisted of 4096 variables. Suppose you want to reduce that to just 2 dimensions, then all you need to do to perform the reduction with PCA is take the dot product between your centered data and the matrix $\boldsymbol{V}=\begin{bmatrix} v_1 & v_2 \end{bmatrix}$, whose columns are the first 2 eigenvectors, or principal components, associated to the 2 largest eigenvalues.
# 
# <a name='ex03'></a>
# ### Exercise 5
# In the next cell you will define a function that, given the data matrix, the eigenvector matrix (always sorted according to decreasing eignevalues), and the number of principal components to use, performs PCA.

# In[337]:


# GRADED cell
def perform_PCA(X, eigenvecs, k):
    """
    Perform dimensionality reduction with PCA
    Inputs:
        X (ndarray): original data matrix. Has dimensions (n_observations)x(n_variables)
        eigenvecs (ndarray): matrix of eigenvectors. Each column is one eigenvector. The k-th eigenvector 
                            is associated to the k-th eigenvalue
        k (int): number of principal components to use
    Returns:
        Xred
    """
    
    ### START CODE HERE ###
    V = eigenvecs[:,:k]
    Xred = np.dot(X, V)
    ### END CODE HERE ###
    return Xred
    # grade-up-to-here


# Try out this function, reducing your data to just two components

# In[338]:


Xred2 = perform_PCA(X, eigenvecs,2)
print(f'Xred2 shape: {Xred2.shape}')


# In[339]:


# Test your solution.
w4_unittest.test_check_PCA(perform_PCA)


# <a name='2.5'></a>
# ### 2.5 Analyzing the dimensionality reduction in 2 dimensions
# 
# One cool thing about reducing your data to just two components is that you can clearly visualize each cat image on the plane. Remember that each axis on this new plane represents a linear combination of the original variables, given by the direction of the two eigenvectors.
# 
# Use the function `plot_reduced_data` in `utils`to visualize the transformed data. Each blue dot represents an image, and the number represents the index of the image. This is so you can later recover which image is which, and gain some intuition.

# In[340]:


utils.plot_reduced_data(Xred2)


# If two points end up being close to each other in this representation, it is expected that the original pictures should be similar as well. 
# Let's see if this is true. Consider for example the images 19, 21 and 41, which appear close to each other on the top center of the plot. Plot the corresponding cat images vertfy that they correspond to similar cats. 

# In[341]:


fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(imgs[19], cmap='gray')
ax[0].set_title('Image 19')
ax[1].imshow(imgs[21], cmap='gray')
ax[1].set_title('Image 21')
ax[2].imshow(imgs[41], cmap='gray')
ax[2].set_title('Image 41')
plt.suptitle('Similar cats')


# As you can see, all three cats have white snouts and black fur around the eyes, making them pretty similar.
# 
# Now, let's choose three images that seem far appart from each other, for example image 18, on the middle right, 41 on the top center and 51 on the lower left, and also plot the images

# In[342]:


fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(imgs[18], cmap='gray')
ax[0].set_title('Image 18')
ax[1].imshow(imgs[41], cmap='gray')
ax[1].set_title('Image 41')
ax[2].imshow(imgs[51], cmap='gray')
ax[2].set_title('Image 51')
plt.suptitle('Different cats')


# In this case, all three cats look really different, one being completely black, another completely white, and the the third one a mix of both colors.
# 
# 
# Feel free to choose different pairs of points and check how similar (or different) the pictures are. 
# 
# <a name='2.6'></a>
# ### 2.6 Reconstructing the images from the eigenvectors
# 
# When you compress the images using PCA, you are losing some information because you are using fewer variables to represent each observation. 
# 
# A natural question arises: how many components do you need to get a good reconstruction of the image? Of course, what determines a "good" reconstruction might depend on the application.
# 
# A cool thing is that with a simple dot product you can transform the data after applying PCA back to the original space. This means that you can reconstruct the original image from the transformed space and check how distorted it looks based on the number of components you kept.
# 
# Suppose you obtained the matrix $X_{red}$ by keeping just two eigenvectors, then $X_{red} = \mathrm{X}\underbrace{\left[v_1\  v_2\right]}_{\boldsymbol{V_2}}$.
# 
# To transform the images back to the original variables space all you need to do is take the dot product between $X_{red}$ and $\boldsymbol{V_2}^T$. If you were to keep more components, say $k$, then simply replace $\boldsymbol{V_2}$ by $\boldsymbol{V_k} = \left[v_1\ v_2\ \ldots\ v_k\right]$. Notice that you can't make any combination you like, if you reduced the original data to just $k$ components, then the recovery must consider only the first $k$ eigenvectors, otherwise you will not be able to perform the matrix multiplication.
# 
# In the next cell you will define a function that given the transformed data $X_{red}$ and the matrix of eigenvectors returns the recovered image. 

# In[343]:


def reconstruct_image(Xred, eigenvecs):
    X_reconstructed = Xred.dot(eigenvecs[:,:Xred.shape[1]].T)

    return X_reconstructed


# Let's see what the reconstructed image looks like for different number of principal components

# In[344]:


Xred1 = perform_PCA(X, eigenvecs,1) # reduce dimensions to 1 component
Xred5 = perform_PCA(X, eigenvecs, 5) # reduce dimensions to 5 components
Xred10 = perform_PCA(X, eigenvecs, 10) # reduce dimensions to 10 components
Xred20 = perform_PCA(X, eigenvecs, 20) # reduce dimensions to 20 components
Xred30 = perform_PCA(X, eigenvecs, 30) # reduce dimensions to 30 components
Xrec1 = reconstruct_image(Xred1, eigenvecs) # reconstruct image from 1 component
Xrec5 = reconstruct_image(Xred5, eigenvecs) # reconstruct image from 5 components
Xrec10 = reconstruct_image(Xred10, eigenvecs) # reconstruct image from 10 components
Xrec20 = reconstruct_image(Xred20, eigenvecs) # reconstruct image from 20 components
Xrec30 = reconstruct_image(Xred30, eigenvecs) # reconstruct image from 30 components

fig, ax = plt.subplots(2,3, figsize=(22,15))
ax[0,0].imshow(imgs[21], cmap='gray')
ax[0,0].set_title('original', size=20)
ax[0,1].imshow(Xrec1[21].reshape(height,width), cmap='gray')
ax[0,1].set_title('reconstructed from 1 components', size=20)
ax[0,2].imshow(Xrec5[21].reshape(height,width), cmap='gray')
ax[0,2].set_title('reconstructed from 5 components', size=20)
ax[1,0].imshow(Xrec10[21].reshape(height,width), cmap='gray')
ax[1,0].set_title('reconstructed from 10 components', size=20)
ax[1,1].imshow(Xrec20[21].reshape(height,width), cmap='gray')
ax[1,1].set_title('reconstructed from 20 components', size=20)
ax[1,2].imshow(Xrec30[21].reshape(height,width), cmap='gray')
ax[1,2].set_title('reconstructed from 30 components', size=20)


# As you can see, as the number of components increases, the reconstructed image looks more and more as the original one. Even with as little as 1 component you can are least identify where the relevant features such as eyes and nose are located. 
# 
# What happens when you consider all of the 55 eigenvectors associated to non-zero eigenvalues? Go ahead and experiment with different number of principal components and see what happens.

# <a name='2.7'></a>
# ### 2.7 Explained variance
# 
# When deciding how many components to use for the dimensionality reduction, one good criteria to consider is the explained variance. 
# 
# The explained variance is measure of how much variation in a dataset can be attributed to each of the principal components (eigenvectors). In other words, it tells us how much of the total variance is “explained” by each component. 
# 
# In PCA, the first principal component, i.e. the eigenvector associated to the largest eigenvalue, is the one with greatest explained variance. As you might remember from the lectures, the goal of PCA is to reduce the dimensionality by projecting data in the directions with biggest variability. 
# 
# In practical terms, the explained variance of a principal component is the ratio between its associated eigenvalue and the sum of all the eigenvalues. So, for our example, if you want the explained variance of the first principal component you will need to do $\frac{\lambda_1}{\sum_{i=1}^{55} \lambda_i}$
# 
# Next, let's plot the explained variance of each of the 55 principal components, or eigenvectors. Don't worry about the fact that you only computed 55 eigenvalue-eigenvector pairs, recall that all the remaining eigenvalues of the covariance matrix are zero, and thus won't add enything to the explained variance.
# 

# In[345]:


explained_variance = eigenvals/sum(eigenvals)
plt.plot(np.arange(1,56), explained_variance)


# As you can see, the explained variance falls pretty fast, and is very small after the 20th component.
# 
# A good way to decide on the number of components is to keep the ones that explain a very high percentage of the variance, for example 95%. 
# 
# For an easier visualization you can plot the cumulative explained variance. You can do this with the `np.cumsum` function. Let's see what this looks like

# In[346]:


explained_cum_variance = np.cumsum(explained_variance)
plt.plot(np.arange(1,56), explained_cum_variance)
plt.axhline(y=0.95, color='r')


# In red you can see the 95% line. This means that if you want to be able to explain 95% of the variance of your data you need to keep 35 principal components. 
# 
# Let's see how some of the original images look after the reconstruction when using 35 principal components 
# 
# 

# In[347]:


Xred35 = perform_PCA(X, eigenvecs, 35) # reduce dimensions to 35 components
Xrec35 = reconstruct_image(Xred35, eigenvecs) # reconstruct image from 35 components

fig, ax = plt.subplots(4,2, figsize=(15,28))
ax[0,0].imshow(imgs[0], cmap='gray')
ax[0,0].set_title('original', size=20)
ax[0,1].imshow(Xrec35[0].reshape(height, width), cmap='gray')
ax[0,1].set_title('Reconstructed', size=20)

ax[1,0].imshow(imgs[15], cmap='gray')
ax[1,0].set_title('original', size=20)
ax[1,1].imshow(Xrec35[15].reshape(height, width), cmap='gray')
ax[1,1].set_title('Reconstructed', size=20)

ax[2,0].imshow(imgs[32], cmap='gray')
ax[2,0].set_title('original', size=20)
ax[2,1].imshow(Xrec35[32].reshape(height, width), cmap='gray')
ax[2,1].set_title('Reconstructed', size=20)

ax[3,0].imshow(imgs[54], cmap='gray')
ax[3,0].set_title('original', size=20)
ax[3,1].imshow(Xrec35[54].reshape(height, width), cmap='gray')
ax[3,1].set_title('Reconstructed', size=20)


# Most of these reconstructions look pretty good, and you were able to save a lot of memory by reducing the data from 4096 variables to just 35!
# 
# Now that you understand how the explained variance works you can play around with different amount of explained variance and see how this affects the reconstructed images. You can also explore how the reconstruction for different images looks. 
# 
# As you can see, PCA is a really useful tool for dimensionality reduction. In this assignment you saw how it works on images, but you can apply the same principle to any tabular dataset. 
# 
# Congratulations! You have finished the assignment in this week.

# 
