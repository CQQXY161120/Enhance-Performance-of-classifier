# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:40:27 2023

@author: Anna
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, metrics
from sklearn.semi_supervised import LabelSpreading
from scipy.spatial.distance import pdist,squareform
from scipy.stats import multivariate_normal
def plot_decision_boundary(model, axis):
    
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    # custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])
    custom_cmap = ListedColormap(['#00FFFF','#D3D3D3'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)
    
np.random.seed(134)

def RLP(Data, Label):
   
    visited, result = set(), []
    n = len(Label)
    Store_Matrix = np.zeros((n,n))
    ### Index matrix that store the neighbor indexs of samples
    
    Sorted_Index = np.argsort(squareform(pdist(Data,'seuclidean'),force='no',checks=True))
    for i in range(n):
        label_i = Label[i]
        for j in range(n):
            if label_i == Label[Sorted_Index[i][j]]:
                Store_Matrix[i][Sorted_Index[i][j]] = 1
            else:
                break
    Store_Matrix = Store_Matrix.astype(int)
    ### Insert reverse neighbor information
    for i in range (n):
        Store_Matrix[i,np.nonzero(Store_Matrix[:,i])[0]] = 1
    ###BFS
    def BFS(Store_Matrix,s):
        queue = []
        queue.append(s) # 向list添加元素，用append()
        seen = set() # 此处为set, python里set用的是hash table, 搜索时比数组要快。
        seen.add(s) # 向set添加函数，用add()
        while (len(queue) > 0):
            vertex = queue.pop(0)  #提取队头
            nodes = np.nonzero(Store_Matrix[vertex,:])  #获得队头元素的邻接元素
            for w in list(nodes)[0]:
                if w not in seen:
                    queue.append(w) #将没有遍历过的子节点入队
                    seen.add(w) #标记好已遍历 
        return seen    
        
    for i in range(n):
        if  i not in visited:
            seen = BFS(Store_Matrix,i)
            visited = visited | set(seen)
            result.append(seen)  
    t = np.log10(n)
    conf = []
    for cl in result:
        if len(cl)>t:
            conf.extend(cl)

    unconf = list(set(list(range(n)))-set(conf))
    label_prop_model = LabelSpreading(kernel='rbf',alpha=0.2)
    label_c = np.copy(Label)
    label_c[unconf] = -1
    label_prop_model.fit(Data, label_c)
    label_p = label_prop_model.predict(Data)
    remain_ind = np.where(Label-label_p==0)[0]

    return remain_ind

def Gaussian_Distribution(N=2, M=1000, m=0, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差
    
    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = [m,0]  
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    
    return data, Gaussian
#%%generated data set
m = 2.0
size = 300
M=500
ls = 35
fs = 40
data1, Gaussian1 = Gaussian_Distribution(N=2, M=M, m=-0.3, sigma=0.2)
data2, Gaussian2 = Gaussian_Distribution(N=2, M=M, m=0.3, sigma=0.2)
x1, y1 = data1.T
x2, y2 = data2.T
data = np.vstack((data1,data2))
label = np.ones((2*M,1)).astype(int)
label[M:] = 2
plt.figure(figsize=(14,10))
plt.scatter(x1,y1, marker='o',c='none',edgecolors='r',s=size,linewidths=2)
plt.scatter(x2,y2, marker='s',c='none',edgecolors='k',s=size,linewidths=2)
plt.xlabel('x', fontsize=fs)
plt.ylabel('y', fontsize=fs)
plt.plot([0,0], [-m,m], 'g--',linewidth = 10)
plt.tick_params(labelsize=ls)
plt.savefig('Generated_dataset.png')
plt.show()
#%%
x_label = ['KNN','DT','SGLB']
classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    CatBoostClassifier(custom_loss=[metrics.Accuracy()],logging_level='Silent',langevin=True)
    ]

###RLP
con_ind = RLP(data, label.ravel())
data_epc, label_epc = data[con_ind,:], label[con_ind]
rl_ind = np.where(label_epc==1)[0]
r2_ind = np.where(label_epc==2)[0]
plt.figure(figsize=(14,10))
plt.scatter(data_epc[rl_ind,0],data_epc[rl_ind,1], marker='o',c='none',edgecolors='r',s=size,linewidths=2)
plt.scatter(data_epc[r2_ind,0],data_epc[r2_ind,1], marker='s',c='none',edgecolors='k',s=size,linewidths=2)
plt.xlabel('x', fontsize=fs)
plt.ylabel('y', fontsize=fs)
plt.plot([0,0], [-m,m], 'g--',linewidth = 10)
plt.tick_params(labelsize=ls)
plt.show()
plt.savefig("Reduced_dataset_RLP.png")

#%%
###Before reduction
i=0
for clf in classifiers:
    clf.fit(data, label)
    plt.figure(figsize=(14,10))
    x1, y1 = data1.T
    x2, y2 = data2.T
    plot_decision_boundary(clf, axis=[-m, m, -m, m])
    plt.plot([0,0], [-m,m], 'g--',linewidth = 10)
    plt.scatter(x1,y1, marker='o',c='none',edgecolors='r',s=size,linewidths=2)
    plt.scatter(x2,y2, marker='s',c='none',edgecolors='k',s=size,linewidths=2)
    plt.xlabel('x', fontsize=fs)
    plt.ylabel('y', fontsize=fs)
    plt.tick_params(labelsize=ls)
    plt.show()
    plt.savefig("%s_before.png"%x_label[i], bbox_inches='tight')
    i+=1
i=0
###RLP
for clf in classifiers:
    plt.figure(figsize=(14,10))
    clf.fit(data_epc, label_epc)
    plot_decision_boundary(clf, axis=[-m, m, -m, m])
    rl_ind = np.where(label_epc==1)[0]
    r2_ind = np.where(label_epc==2)[0]
    plt.scatter(data_epc[rl_ind,0],data_epc[rl_ind,1], marker='o',c='none',edgecolors='r',s=size,linewidths=2)
    plt.scatter(data_epc[r2_ind,0],data_epc[r2_ind,1], marker='s',c='none',edgecolors='k',s=size,linewidths=2)
    plt.xlabel('x', fontsize=fs)
    plt.ylabel('y', fontsize=fs)
    plt.plot([0,0], [-m,m], 'g--',linewidth = 10)
    plt.tick_params(labelsize=ls)
    plt.show()
    plt.savefig("%s_after_RLP.png"%x_label[i], bbox_inches='tight')
    i+=1