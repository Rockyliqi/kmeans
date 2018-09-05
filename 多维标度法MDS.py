
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
iris = load_iris().data


# In[2]:


import numpy as np
def dist_matrix(dataset):#生成一个距离矩阵
    m,n = dataset.shape#获取数据集的行列
    matrix = np.zeros((m,m))#生成一个全零的方阵
    for i in range(m):
        for j in range(i+1,m):
            dist = dataset[i,:]-dataset[j,:]#获取数据集的第i,j样本计算距离
            distance = np.dot(dist,dist.T)#矩阵的乘法
            matrix[i,j] = matrix[j,i] = distance#依次进行赋值
    return matrix


# In[5]:


import matplotlib.pyplot as plt
def MDS(matrix):
    D = matrix #传入一个距离矩阵
    M,N  = matrix.shape#获取距离矩阵的行和列
                #   D = dist_matrix(iris)
                #M,N = dist_matrix(iris).shape
    T = np.zeros((N,N))#生成一个全零方阵用来存储
    #k = np.dot(D,D.T)#矩阵乘法或者np.transpose(D)
    H = np.eye(N)-1/N#np.eye生成一个单位矩阵
    T = -0.5*np.dot(np.dot(H,D),H)#分别对距离矩阵的行列进行去均值
    eigVal,eigVec = np.linalg.eig(T)#计算矩阵特征值特征向量
    X = np.dot(eigVec[:,:2],np.diag(np.sqrt(eigVal[:2])))#获取矩阵T特征向量的前两个，特征值的前两个构成对角阵进行相乘
    return X

if __name__ == '__main__':
    matrix = dist_matrix(iris)
    X = MDS(matrix)
    plt.scatter(X[:,0],X[:,1],c='black')
    plt.show()

