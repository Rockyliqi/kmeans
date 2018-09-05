
# coding: utf-8

# In[1]:


def mashi_distance(a,b,dataset):#马氏距离
    #a = np.array(a)
    #b = np.array(b)
    v = a-b
    cov = covs(dataset)
    mashi = np.dot(np.dot(v,cov),v.T)#
    mashi_ = np.sqrt(mashi)
    return mashi_
#获取数据集的协方差矩阵
def covs(dataset):
    m,n = dataset.shape
    i_list = []
    cov = np.zeros((n,m))
    for i in range(n):
        mean = sum(dataset[:,i])/m
        i_list.append(mean)
        means = np.array(i_list)#获取变量的均值
    for j in range(m):
        for k in range(n):
            cov[k,j] = dataset[j,k]-means[k]
    covs = np.dot(cov,cov.T)/m#获取数据集的协方差矩阵
    return np.linalg.inv(covs)#协方差矩阵逆阵


# In[2]:


def seulid_distance(a,b,dataset):#标准化后的欧氏距离
    a = np.array(a)
    b = np.array(b)
    v = a-b
    cov = covs(dataset)
    mashi = np.dot(np.dot(v,cov),v.T)
    mashi_ = np.sqrt(mashi)
    return mashi_
def covs(dataset):
    m,n = dataset.shape
    i_list = []
    cov = np.zeros((n,m))
    s_eulid = np.zeros((n,n))
    for i in range(n):
        mean = sum(dataset[:,i])/m
        i_list.append(mean)
        means = np.array(i_list)#获取变量的均值
    for j in range(m):
        for k in range(n):
            cov[k,j] = dataset[j,k]-means[k]
    covs = np.dot(cov,cov.T)/m#获取数据集的协方差矩阵
    covi = np.linalg.inv(covs)
    for z in range(len(covi)):
        s_eulid[z,z] = covi[z,z]
    return s_eulid#获取一个对角矩阵


# In[3]:


def distancenorm(Norm,a,b):
    D_value = a - b
    if Norm == '1':
        count = np.absolute(D_value)#绝对值距离
        dist = np.sum(count)
    elif Norm == '2':#欧氏距离
        count = np.power(D_value,2)
        dist = np.sqrt(sum(count))
    elif Norm == 'Infinity':#切比雪夫距离
        count = np.absolute(D_value)
        dist = np.max(count)
    return dist


# In[ ]:


def lance(a,b):
    #兰氏距离
    count1 = np.absolute(a-b)
    count2 = a+b
    dist = sum(count1/count2)/len(a)
    return dist

