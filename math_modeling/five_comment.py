# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 16:51:28 2022

@author: zklei
"""
# In[]层次分析
import numpy as np  
A=np.array([[1,1/3,1/4,1/5],
            [3,1,3/4,3/5],
            [4,4/3,1,4/5],
            [5,5/3,5/4,1]])
m=len(A)
n=len(A[0])
RI=[0,0,0.58,0.90,1.12,1.24,1.32,1.41,1.45,1.49,1.51]
R=np.linalg.matrix_rank(A)#求判断矩阵的秩
V,D=np.linalg.eig(A)
list1=list(V)
B=np.max(list1)#最大特征值
index=list1.index(B)
C=D[:,index]#对应向量
CI=(B-n)/(n-1)
CR=CI/RI[n]
if CR<0.1:
    print("CI:",CI)
    print("CR:",CR)
    print("通过一致性检验!,各个权重向量为:")
    sum=np.sum(C)
    Q=C/sum
    print(Q)
else:
    print("未通过一致性检验!请再次构造!")


# In[]
def ministand(datas,offset=0):
    def normalization(data):
        return (np.max(data)-data)/(np.max(data)-np.min(data))
    return list(map(normalization, datas))

def middlestand(datas,x_min,x_max):
    def normalization(data):
        if data<=x_min or data>=x_max:
            return 0
        elif data >x_min and data<(x_min+x_max)/2:
            return 2 * (data - x_min) / (x_max - x_min)
        elif data<x_max and data>=(x_min+x_max)/2:
            return 2 * (x_max-data) / (x_max-x_min)
    return list(map(normalization, datas))


def qujian(datas,x_min,x_max,x_mininum,x_maxinum):
    def normalization(data):
        if data >=x_min and data<=x_max:
            return 1
        elif data<=x_mininum or data >=x_maxinum:
            return 0
        elif data>x_max and data <x_maxinum:
            return 1-(data-x_max)/(x_maxinum-x_max)
        elif data <x_min and data >x_mininum:
            return 1-(x_min-data)/(x_min-x_mininum)
    return list(map(normalization,datas))

import pandas as pd
import numpy as np
def entropyWeight(data):
	data = np.array(data)
	# 归一化
	P = data / data.sum(axis=0)#(data-np.min(data,axis=0)) / (np.max(data,axis=0)-np.min(data,axis=0))
	# 计算熵值#data / data.sum(axis=0)
	E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)
	# 计算权系数
	return (1 - E) / (1 - E).sum()

def topsis(data,weight=None):
    #归一化
    data=data/np.sqrt((data**2).sum())
    #最优最劣方案
    imagination = pd.DataFrame([data.min(), data.max()], index=['负理想解', '正理想解'])
    weight = entropyWeight(data) if weight is None else np.array(weight)
    result=data.copy()
    result['正理想解']=np.sqrt(((data-imagination.loc['正理想解'])**2*weight).sum(axis=1))
    result['负理想解']=np.sqrt(((data-imagination.loc['负理想解'])**2*weight).sum(axis=1))

    result['综合得分']=result['负理想解']/(result['负理想解']+result['正理想解'])
    result['排序']=result.rank(ascending=False)['综合得分']

    return result, imagination, weight
# In[]
data = pd.DataFrame(
    {'人均专著': [0.1, 0.2, 0.4, 0.9, 1.2], '生师比': [5, 6, 7, 10, 2], '科研经费': [5000, 6000, 7000, 10000, 400],
     '逾期毕业率': [4.7, 5.6, 6.7, 2.3, 1.8]}, index=['院校' + i for i in list('ABCDE')])

data['生师比'] = qujian(data['生师比'], 5, 6, 2, 12) 
data['逾期毕业率'] = ministand(data['逾期毕业率'])
out = topsis(data)    # 也可以设置权系数
# In[]
#rsr模糊评价可以
import sys
sys.path.append("C:/Users/zklei/Pictures/Camera Roll")
import pandas as pd
from rsr import rsr,rsrAnalysis
data = pd.DataFrame({'产前检查率': [99.54, 96.52, 99.36, 92.83, 91.71, 95.35, 96.09, 99.27, 94.76, 84.80],
                     '孕妇死亡率': [60.27, 59.67, 43.91, 58.99, 35.40, 44.71, 49.81, 31.69, 22.91, 81.49],
                     '围产儿死亡率': [16.15, 20.10, 15.60, 17.04, 15.01, 13.93, 17.43, 13.89, 19.87, 23.63]},
                    index=list('ABCDEFGHIJ'), columns=['产前检查率', '孕妇死亡率', '围产儿死亡率'])
data["孕妇死亡率"] = 1 / data["孕妇死亡率"]
data["围产儿死亡率"] = 1 / data["围产儿死亡率"]
[Result, Distribution]=rsr(data)
# In[]
#数据包络
import sys
sys.path.append("C:/Users/zklei/Pictures/Camera Roll")
from dea import DEA
if __name__ == "__main__":
    X = np.array([
        [20., 300.],
        [30., 200.],
        [40., 100.],
        [20., 200.],
        [10., 400.]
    ])
    y = np.array([
        [1000.],
        [1000.],
        [1000.],
        [1000.],
        [1000.]
    ])
    dea = DEA(X,y)
    rs = dea.fit()
    print(rs)
# In[]
import numpy as np
import xlrd
import pandas as pd
def dataDirection_1(d):         
		return np.max(d)-d#极小型指标转化为极大型指标

def dataDirection_2(d, x_best):
    temp_datas = d - x_best
    M = np.max(abs(temp_datas))
    answer_datas = 1 - abs(d - x_best) / M#中间型指标转化为极大型指标
    return answer_datas

    
def dataDirection_3(d, x_min, x_max):
    M = max(x_min - np.min(d), np.max(d) - x_max)
    answer_list = []
    for i in d:
        if(i < x_min):
            answer_list.append(1 - (x_min-i) /M)
        elif( x_min <= i <= x_max):
            answer_list.append(1)
        else:
            answer_list.append(1 - (i - x_max)/M)
    return np.array(answer_list)#区间型指标转化为极大型指标   
 
#正向化矩阵标准化
def temp2(d):
    K = np.power(np.sum(pow(d,2),axis =1),0.5)
    for i in range(0,K.size):
        for j in range(0,d[i].size):
            d[i,j] = d[i,j] / K[i]#套用矩阵标准化的公式
    return d

#计算得分并归一化
def temp3(answer2):
    list_max = np.array([np.max(answer2[0,:]),np.max(answer2[1,:]),np.max(answer2[2,:]),np.max(answer2[3,:])])  #获取每一列的最大值
    list_min = np.array([np.min(answer2[0,:]),np.min(answer2[1,:]),np.min(answer2[2,:]),np.min(answer2[3,:])])  #获取每一列的最小值
    max_list = []       #存放第i个评价对象与最大值的距离
    min_list = []       #存放第i个评价对象与最小值的距离
    answer_list=[]      #存放评价对象的未归一化得分
    for k in range(0,np.size(answer2,axis = 1)):        #遍历每一列数据
        max_sum = 0
        min_sum = 0
        for q in range(0,4):                                #有四个指标
            max_sum += np.power(answer2[q,k]-list_max[q],2)     #按每一列计算Di+
            min_sum += np.power(answer2[q,k]-list_min[q],2)     #按每一列计算Di-
        max_list.append(pow(max_sum,0.5))
        min_list.append(pow(min_sum,0.5))
        answer_list.append(min_list[k]/ (min_list[k] + max_list[k]))
        #套用计算得分的公式 Si = (Di-) / ((Di+) +(Di-))
        max_sum = 0
        min_sum = 0
    answer = np.array(answer_list)      #得分归一化
    return (answer / np.sum(answer))

answer1 = np.array(pd.read_excel('E:/机器学习方法/topsis.xlsx'))
answer2 = []
for i in range(0, 4):       #按照不同的列，根据不同的指标转换为极大型指标，因为只有四列
    answer = None
    if(i == 0):             #本来就是极大型指标，不用转换
        answer = answer1[:,0]             
    elif(i == 1):                   #中间型指标
        answer = dataDirection_2(answer1[:,1],7)
    elif(i==2):                     #极小型指标
        answer = dataDirection_1(answer1[:,2])
    else:                           #范围型指标
        answer = dataDirection_3(answer1[:,3],10,20)
    answer2.append(answer)
answer2 = np.array(answer2)         #将list转换为numpy数组
answer3 = temp2(answer2)            #数组正向化
answer4 = temp3(answer3)            #标准化处理去钢
data = pd.DataFrame(answer4)        #计算得分
# In[]
# 导入相关包和数据
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# 初始化数据集
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
df['label'] = cancer.target
df.head()
# In[]
# 在做因子分析之前, 我们需要先做充分性检测, 就是数据集中是否能找到这些factor, 我们可以使用下面的两种方式进行寻找。
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity,calculate_kmo
chi_square_value, p_value = calculate_bartlett_sphericity(df)
print(chi_square_value,p_value)
#巴莱特球形检验,p值小于0.05
# In[]
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all, kmo_model = calculate_kmo(df)
print(kmo_model)
#KMO值大于0.7
# In[]
fa = FactorAnalyzer(31,rotation='varimax')
fa.fit(df)
ev,v = fa.get_eigenvalues()

# 可视化
# plot横轴是指标个数，纵轴是ev值
# scatter横轴是指标个数，纵轴是ev值

plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
# In[]
fa = FactorAnalyzer(10, rotation="varimax")
fa.fit(df)

# # 31*5(变量个数*因子个数)
print(fa.loadings_.shape)

# In[]
import seaborn as sns
df_cm = pd.DataFrame(np.abs(fa.loadings_),index=df.columns)

fig,ax = plt.subplots(figsize=(12,10))
sns.heatmap(df_cm,annot=True,cmap='BuPu',ax=ax)
# 设置y轴字体的大小
ax.tick_params(axis='x',labelsize=15)
ax.set_title("Factor Analysis",fontsize=12)
ax.set_ylabel("Sepal Width")

# In[]
e=pd.DataFrame(fa.transform(df))
# In[]
#以下为一个完整的因子分析流程
# In[]
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# 皮尔森相关系数
std = StandardScaler()
data_zs = std.fit_transform(df)
data_corr=df.corr()
print("\n相关系数:\n",data_corr)
#获得协方差矩阵，cov是numpy库中计算协方差的函数，获得协方差矩阵9*9
#注：标准化后的矩阵的协方差矩阵 即为 原始数据的相关系数矩阵
data_zs_cov = np.cov(data_zs.T)
print("\n协方差矩阵：\n",data_zs_cov)
# In[]
#求解特征值以及特征向量，直接调用求特征值和特征向量的函数
import numpy.linalg as nlg
e , ev = nlg.eig(data_zs_cov)
#查看特征值
eig = pd.DataFrame()
eig['names'] = df.columns
eig['e'] = e
eig.sort_values('e', ascending=False, inplace=True)
print("\n特征值\n：",eig)
#查看特征向量，一行为一个特征向量
eig1=pd.DataFrame(ev)
eig1.columns = df.columns
eig1.index = df.columns
print("\n特征向量\n",eig1)

# In[]
#计算各特征值的方差贡献率
w = list()
for i in range(len(e)):
    w.append(eig['e'][i] / eig['e'].sum())
print(w)
#计算特征值的累计贡献率
q = list()
for j in range(len(e)):
    q.append(eig['e'][:j].sum() / eig['e'].sum())
print(q)
# 求公因子个数m,使用前m个特征值的比重大于80%的标准，选出了公共因子,根据累计贡献率得出
for m in range(len(e)):
    print(eig['e'][:m].sum())
    print(eig['e'].sum())
    print(eig['e'][:m].sum() / eig['e'].sum())
    if eig['e'][:m].sum() / eig['e'].sum() >= 0.8:
        print("\n主成分个数:", m)
        break
# In[]
#主成分个数为m个,提取方法为主成分提取方法,旋转方法为最大方差法,
fa = FactorAnalyzer(n_factors=m , method='principal' , rotation='varimax')
fa.fit(df)
#因子载荷矩阵(成分矩阵)
print(pd.DataFrame(fa.loadings_))

# In[]
import seaborn as sns
df_cm = pd.DataFrame(np.abs(fa.loadings_),index=df.columns)

fig,ax = plt.subplots(figsize=(12,10))
sns.heatmap(df_cm,annot=True,cmap='BuPu',ax=ax)
# 设置y轴字体的大小
ax.tick_params(axis='x',labelsize=15)
ax.set_title("Factor Analysis",fontsize=12)
ax.set_ylabel("Sepal Width")
# In[]
# 给出贡献率,第一行表示特征值方差，第二行表示贡献率，第三行表示累计贡献率
#该过程与上述求方差贡献率结果一致
var = fa.get_factor_variance()  
for i in range(0,3):
    print(var[i])
#公因子方差 ,特殊因子方差，因子的方差贡献度 ，反映公共因子对所有变量的贡献
print(fa.get_communalities())
# In[]
#计算因子得分
fa_t_score = np.dot(np.mat(data_zs), np.mat(fa.loadings_))
print("\n每个企业的因子得分：\n",pd.DataFrame(fa_t_score))
# In[]
#综合得分(加权计算）
weight = var[1]     #计算每个因子的权重
fa_t_score_final = ((np.dot(fa_t_score, weight) / e.sum()).real).reshape(569)
f_score=np.zeros(569)
for i in range(569):
    f_score[i]=fa_t_score_final[0,i]
# In[]
plt.plot(np.arange(1,fa_t_score_final.shape[1]+1,1),f_score)
# In[]
from sklearn.decomposition import PCA  # 导入 sklearn.decomposition.PCA 类
import numpy as np  # Youcans， XUPT

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
modelPCA = PCA(n_components=2)  # 建立模型，设定保留主成分数 K=2
modelPCA.fit(X)  # 用数据集 X 训练 模型 modelPCA
print("返回 PCA 模型保留的主成份个数:",modelPCA.n_components_)  # 返回 PCA 模型保留的主成份个数
# 2
print("返回 PCA 模型各主成份占比:",modelPCA.explained_variance_ratio_)  # 返回 PCA 模型各主成份占比
# [0.9924 0.0075]  # print 显示结果
print("返回 PCA 模型各主成份的奇异值:",modelPCA.singular_values_) # 返回 PCA 模型各主成份的奇异值
# [6.3006 0.5498]  # print 显示分类结果

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
modelPCA2 = PCA(n_components=0.9) # 建立模型，设定主成份方差占比 0.9
# 用数据集 X 训练 模型 modelPCA2，并返回降维后的数据
Xtrans = modelPCA2.fit_transform(X)

print("返回 PCA 模型保留的主成份个数:",modelPCA2.n_components_)  # 返回 PCA 模型保留的主成份个数
# 1
print("返回 PCA 模型各主成份占比:",modelPCA2.explained_variance_ratio_)  # 返回 PCA 模型各主成份占比
# [0.9924]  # print(Youcans) 显示结果 
print(" PCA 模型各主成份占比:",modelPCA2.singular_values_)  # 返回 PCA 模型各主成份占比
# [6.3006]  # print 显示结果 
print(Xtrans)  # 返回降维后的数据 Xtrans
# [[1.3834], [2.2219], [3.6053], [-1.3834], [-2.2219], [-3.6053]]

# In[]
# Demo of sklearn.decomposition.IncrementalPCA
#大样本，特征量时使用
from sklearn.datasets import load_digits
from sklearn.decomposition import IncrementalPCA, PCA
from scipy import sparse  # Youcans， XUPT

X, _ = load_digits(return_X_y=True)
print(type(X))  # <class 'numpy.ndarray'>
print(X.shape)      # (1797, 64)

modelPCA = PCA(n_components=6)  # 建立模型，设定保留主成分数 K=6
modelPCA.fit(X)  # 用数据集 X 训练 模型 modelPCA
print(modelPCA.n_components_)  # 返回 PCA 模型保留的主成份个数
# 6
print(modelPCA.explained_variance_ratio_)  # 返回 PCA 模型各主成份占比
# [0.1489 0.1362 0.1179 0.0841 0.0578 0.0492]
print(sum(modelPCA.explained_variance_ratio_))  # 返回 PCA 模型各主成份占比
# 0.5941
print(modelPCA.singular_values_) # 返回 PCA 模型各主成份的奇异值
# [567.0066  542.2518 504.6306 426.1177 353.3350 325.8204]

# let the fit function itself divide the data into batches
Xsparse = sparse.csr_matrix(X)  # 压缩稀疏矩阵，并非 IPCA 的必要步骤
print(type(Xsparse))  # <class 'scipy.sparse.csr.csr_matrix'>
print(Xsparse.shape)  # (1797, 64)
modelIPCA = IncrementalPCA(n_components=6, batch_size=200)
modelIPCA.fit(Xsparse)  # 训练模型 modelIPCA

print(modelIPCA.n_components_)  # 返回 PCA 模型保留的主成份个数
# 6
print(modelIPCA.explained_variance_ratio_)  # 返回 PCA 模型各主成份占比
# [0.1486 0.1357 0.1176 0.0838 0.0571 0.0409]
print(sum(modelIPCA.explained_variance_ratio_))  # 返回 PCA 模型各主成份占比
# 0.5838
print(modelIPCA.singular_values_) # 返回 PCA 模型各主成份的奇异值
#[566.4544 541.334 504.0643 425.3197 351.1096 297.0412]

# In[]
# Demo of sklearn.decomposition.KernelPCA

from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA, PCA
import matplotlib.pyplot as plt
import numpy as np  # Youcans， XUPT

X, y = load_iris(return_X_y=True)
print(type(X))  # <class 'numpy.ndarray'>

modelPCA = PCA(n_components=2)  # 建立模型，设定保留主成分数 K=2
Xpca = modelPCA.fit_transform(X)  # 用数据集 X 训练 模型 modelKPCA

modelKpcaP = KernelPCA(n_components=2, kernel='poly') # 建立模型，核函数：多项式
XkpcaP = modelKpcaP.fit_transform(X)  # 用数据集 X 训练 模型 modelKPCA

modelKpcaR = KernelPCA(n_components=2, kernel='rbf') # 建立模型，核函数：径向基函数
XkpcaR = modelKpcaR.fit_transform(X)  # 用数据集 X 训练 模型 modelKPCA

modelKpcaS = KernelPCA(n_components=2, kernel='cosine') # 建立模型，核函数：余弦函数
XkpcaS = modelKpcaS.fit_transform(X)  # 用数据集 X 训练 模型 modelKPCA

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
for label in np.unique(y):
    position = y == label
    ax1.scatter(Xpca[position, 0], Xpca[position, 1], label='target=%d' % label)
    ax1.set_title('PCA')
    ax2.scatter(XkpcaP[position, 0], XkpcaP[position, 1], label='target=%d' % label)
    ax2.set_title('kernel= Poly')
    ax3.scatter(XkpcaR[position, 0], XkpcaR[position, 1], label='target=%d' % label)
    ax3.set_title('kernel= Rbf')
    ax4.scatter(XkpcaS[position, 0], XkpcaS[position, 1], label='target=%d' % label)
    ax4.set_title('kernel= Cosine')
plt.suptitle("KernalPCA(Youcans，XUPT)")
plt.show()