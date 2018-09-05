
# coding: utf-8

# In[1]:


#斜角空间距离
def mean(data):
    m,n = data.shape
    means = np.zeros(n)
    var = np.zeros(n)
    for i in range(n):
        means[i] = sum(data[:,i])/m
    return means 
def varss(data):
    z,y = data.shape
    means = mean(data)
    var = np.zeros(y)
    for i in range(y):
        var[i] = (sum(data[:,i]-means[i])**2)/z
    return var
def covs(data):
    u,v = data.shape
    means = mean(data)
    cov = np.zeros((v,u))
    for i in range(u):#m=150
        for j in range(v):#n=4
            cov[j,i] = data[i,j]-means[j]
    COV = np.dot(cov,cov.T)/u
    return COV
def corr(data):
    cov = covs(data)
    var = varss(data)
    p,q = cov.shape
    corr = np.zeros((p,q))
    for i in range(p):
        for j in range(q):
            corr[i,j] = cov[i,j]/np.sqrt(var[i]*var[j])
    return corr
def x_distance(a,b,dataset):
    coor = corr(dataset)
    x = a-b
    dis = np.zeros(len(x))
    for i in range(len(x)):
        dis[i] = np.dot(np.dot(x,x[i]),coor[:,i])
        dist = np.sqrt(sum(dis)/len(x)**2)
    return  dist


# In[2]:


def distancenorm(Norm,a,b):
    D_value = a - b
    if Norm == '1':#绝对值距离
        count = np.absolute(D_value)
        dist = np.sum(count)
    elif Norm == '2':#欧氏距离
        count = np.power(D_value,2)
        dist = np.sqrt(sum(count))
    elif Norm == 'Infinity':#切比雪夫距离
        count = np.absolute(D_value)
        dist = np.max(count)
    return dist

def calcuDistance(vec1, vec2):
    # 计算向量vec1和向量vec2之间的欧氏距离
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def euclDistance(vector1, vector2):#欧氏距离
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))

def mashi_distance(a,b,dataset):#马氏距离
    #a = np.array(a)
    #b = np.array(b)
    v = a-b
    cov = covs(dataset)
    mashi = np.dot(np.dot(v,cov),v.T)
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

def lance(a,b):
    #兰氏距离
    count1 = np.absolute(a-b)
    count2 = a+b
    dist = sum(count1/count2)/len(a)
    return dist

def getVar(clusterDict, centroidList):
    # 计算簇集合间的均方误差
    # 将簇类中各个向量与质心的距离进行累加求和
    sum = 0.0
    for key in clusterDict.keys():#获取字典的键
        vec1 = np.array(centroidList[key])#字典的键用于索引中心点
        distance = 0.0
        for item in clusterDict[key]:
            vec2 = np.array(item)
            distance += x_distance(vec1, vec2,data)
        sum += distance
    return sum

def minDistance(dataSet, centroidList):#参数数据集和中心点列表
    # 对每个属于dataSet的item，计算item与centroidList中k个质心的欧式距离，找出距离最小的，
    # 并将item加入相应的簇类中
    clusterDict = dict()                 # 用dict来保存簇类结果
    for item in dataSet:
        vec1 = np.array(item)         # 转换成array形式
        flag = 0                         # 簇分类标记，记录与相应簇距离最近的那个簇
        minDis = float("inf")            # 初始化为最大值
        for i in range(len(centroidList)):
            vec2 = np.array(centroidList[i])
            distance = x_distance(vec1,vec2,data)  # 计算相应的欧式距离
            if distance < minDis:    
                minDis = distance
                flag = i                          # 循环结束时，flag保存的是与当前item距离最近的那个簇标记
        if flag not in clusterDict.keys():   # 簇标记不存在，进行初始化
            clusterDict[flag] = list()      
        clusterDict[flag].append(item)       # 加入相应的类别中
    return clusterDict # 返回新的聚类结果
    
def getCentroids(clusterDict):
    # 得到k个质心
    centroidList = list()
    for key in clusterDict.keys():
        centroid = np.mean(np.array(clusterDict[key]),axis=0)#clusterDict[key]列表下的数组然后转换为数组，计算每列的均值，即找到质心
        #axis=0计算列
        centroidList.append(centroid)
    return np.array(centroidList).tolist()#tolist数组转化为列表

def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape#获取数据的维度
    centroids = np.zeros((k, dim))#初始一个k*dim的0矩阵
    for i in range(k):
        index = int(random.uniform(0, 4))#生成一个均匀分布的随机数取整
        centroids[i,:] = dataSet[index,:]#用这个随机数选取数据集的某一点作为中心点
    return centroids#返回随机选取的中心


def showCluster(centroidList, clusterDict):
    # 展示聚类结果
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow']      # 不同簇类的标记 'or' --> 'o'代表圆，'r'代表red，'b':blue
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']   # 质心标记 同上'd'代表棱形
    for key in clusterDict.keys():
        plt.plot(centroidList[key][0], centroidList[key][1], centroidMark[key], markersize = 12)  # 画质心点
        for item in clusterDict[key]:
            plt.plot(item[0], item[1], colorMark[key]) # 画簇类下的点
    plt.show()


# In[3]:


from sklearn import datasets
import matplotlib.pyplot as plt
from numpy import random
import numpy as np
iris = datasets.load_iris()
data = iris.data
centroidList = initCentroids(data, 3)
clusterDict = minDistance(data,centroidList)
newVar = getVar(clusterDict, centroidList) 
#clusterDict
        


# In[164]:


centroidList = initCentroids(data, 3)
clusterDict = minDistance(data,centroidList)
newVar = getVar(clusterDict, centroidList)         # 获得均方误差值，通过新旧均方误差来获得迭代终止条件
oldVar = 0.0001 
showCluster(centroidList,clusterDict)
print('***** 第1次迭代 *****')
for key in clusterDict.keys():
    #print (key, ' --> ', clusterDict[key])
    print ('k个均值向量: \n', centroidList)
    print ('平均均方误差: ', newVar)
    showCluster(centroidList, clusterDict)             # 展示聚类结果
    k = 2
    while abs(newVar - oldVar) >= 0.0001:              # 当连续两次聚类结果小于0.0001时，迭代结束          
        centroidList = getCentroids(clusterDict)          # 获得新的质心
        clusterDict = minDistance(data, centroidList)  # 新的聚类结果
        oldVar = newVar                                   
        newVar = getVar(clusterDict, centroidList)
        print('***** 第%d次迭代 *****' % k)
        #print ('簇类')
        #for key in clusterDict.keys():
            #print( key, ' --> ', clusterDict[key])
        print ('k个均值向量: \n', centroidList)
        print ('平均均方误差: \n', newVar)
        showCluster(centroidList, clusterDict)            # 展示聚类结果
        k += 1


# In[165]:


#欧氏距离下平均均方误差:  97.34621969415682
#马氏距离下平均均方误差:  214.45621308685764
#标准欧氏距离平均均方误差:  341.64432304874526
#兰氏距离下平均均方误差：7.91566381781145
#绝对值距离均方误差：164.0120384153661
#切比雪夫距离均方误差：91.82711598746084
#斜交错距离均方误差：7508917967957760.0


# In[166]:


#for i in range(len(clusterDict[2])):
label = np.zeros(150)
a = len(clusterDict[0])
b = len(clusterDict[1])
c = len(clusterDict[2])
for i in range(a):
    label[i]=0
for j in range(a,a+b):
    label[j]=1
for k in range(a+b,a+b+c):
    label[k]=2
label


# In[167]:


from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
Y = iris.target

def plot_confusion_matrix(y_true, y_pred, labels):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,4), dpi=80)#规定画面大小
    ind_array = np.arange(len(labels))#返回以个数组
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #

        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')

        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                #这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
    if(intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title('')
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels,rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True Classes')
    plt.xlabel('Predict Classes')
    plt.savefig('confusion_matrix.jpg', dpi=300)
    plt.show()
    
labels = [0,1,2]
plot_confusion_matrix(Y,label,labels)

