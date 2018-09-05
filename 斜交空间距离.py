
# coding: utf-8

# In[1]:


from sklearn import datasets
dataset = datasets.load_iris()
data = dataset.data


# In[2]:


import numpy as np
def mean(data):#返回变量均值
    m,n = data.shape
    means = np.zeros(n)
    var = np.zeros(n)
    for i in range(n):
        means[i] = sum(data[:,i])/m
    return means
mean(data)


# In[3]:


def varss(data):#返回变量方差
    m,n = data.shape
    means = mean(data)
    var = np.zeros(n)
    for i in range(n):
        var[i] = (sum(data[:,i]-means[i])**2)/m
    return var


# In[4]:


data.shape


# In[5]:


def covs(data):#返回变量协方差
    m,n = data.shape
    means = mean(data)
    cov = np.zeros((n,m))
    for i in range(m):#m=150
        for j in range(n):#n=4
            cov[j,i] = data[i,j]-means[j]
    COV = np.dot(cov,cov.T)/m
    return COV
covs(data)


# In[6]:


def corr(data):#返回变量相关系数阵
    cov = covs(data)
    var = varss(data)
    m,n = cov.shape
    corr = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            corr[i,j] = cov[i,j]/np.sqrt(var[i]*var[j])
    return corr



# In[8]:


def x_distance(a,b,data):#斜交错空间距离
    coor = corr(data)
    x = a-b
    dis = np.zeros(len(x))
    for i in range(len(x)):
        dis[i] = np.dot(np.dot(x,x[i]),coor[:,i])
        dist = np.sqrt(sum(dis)/len(x)**2)
    return  dist

