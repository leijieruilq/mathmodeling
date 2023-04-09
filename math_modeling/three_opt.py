# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:54:25 2022

@author: zklei
"""

# In[]
# =============================================================================
# #method:
#     - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
#     - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
#     - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
#     - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
#     - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
#     - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
#     - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
#     - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
#     - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
#     - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
#     - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
#     - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
#     - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
#     - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
# =============================================================================
import numpy as np
from scipy.optimize import minimize

def rosen(x):
    return sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)

x0=np.array([1.3,0.7,0.8,1.9,1.2])
res = minimize(rosen, x0, method='Powell',options={'xatol': 1e-8, 'disp': True})
print(res.x)
# In[]线性回归实战
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
x=np.array([825,215,1070,550,480,920,1350,325,670,1215])
y=np.array([3.5,1,4,2,1,3,4.5,1.5,3,5])
import seaborn as sns
sns.set(style='darkgrid')
plt.scatter(x,y,edgecolors="green")
plt.grid(True)
plt.show()
# 计算相关系数
print(np.corrcoef(x,y))
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
X=sm.add_constant(x)
model=sm.OLS(y,X)
result=model.fit()
print(result.summary())
yfit=result.fittedvalues
print(result.params)
print("截距为:b=",result.params[0],"系数为:a=",result.params[1])
prstd,ivlow,ivup=wls_prediction_std(result)#返回标准偏差和置信区间
from matplotlib import pyplot as plt
fig,ax=plt.subplots(figsize=(10,8))
ax.plot(x,y,'o',label="data")
ax.plot(x,yfit,'r-',label="OLS")
ax.plot(x,ivup,'--',color='orange',label='upconf')
ax.plot(x,ivlow,'--',color='orange',label='lowconf')
ax.legend(loc='best')#显示图例
plt.title('OLS linear regression for Event Promotion')
plt.grid(True)
plt.show()
#求随机误差的方差估计
print(f"误差的方差估计为{sum((yfit-y)**2)/(x.shape[0]-2)}")
#print(sum((yfit-y)**2)/(x.shape[0]-2))
# 模型的拟合优度可决系数
print(sum((yfit-np.mean(y))**2)/sum((y-np.mean(y))**2))
#计算x与y的决定系数
print(f"x与y的决定系数为{1-(sum((yfit-y)**2)/(x.shape[0]-1-1))/(sum((y-np.mean(y))**2)/(x.shape[0]-1))}")
print(1-(sum((yfit-y)**2)/(x.shape[0]-1-1))/(sum((y-np.mean(y))**2)/(x.shape[0]-1)))
from scipy import stats 
print(stats.f_oneway(x,y))
#p-value<0.05拒绝假设原假设不同学历对收入影响无差异不成立。
# 残差图
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
sns.residplot(x,y)
plt.title("残差图")
plt.xlabel("张数")
plt.ylabel("残差值")
plt.show()
# 预测x=1000
y_1000_predict=result.params[0]+1000*result.params[1]
print("预测值为:",y_1000_predict)
# In[]
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
x1=np.array([70,75,65,74,72,68,78,66,70,65])
x2=np.array([35,40,40,42,38,45,42,36,44,42])
x3=np.array([1,2.4,2,3,1.2,1.5,4,2,3.2,3])
y=np.array([160,260,210,265,240,220,275,160,275,250])
import pandas as pd
#相关系数之间密切程度
dat=pd.DataFrame({'x1':x1,'x2':x2,'x3':x3})
print(dat.corr().round(4))
import scipy.stats as st
#相关系数检验
print(st.pearsonr(x1,x2))
print(st.pearsonr(x2,x3))
print(st.pearsonr(x1,x3))
#偏相关系数检验
import pingouin as pg
print(pg.partial_corr(data=dat,x='x1',y='x2',covar='x3'))#p<0.01
print(pg.partial_corr(data=dat,x='x1',y='x3',covar='x2'))#p<0.01
print(pg.partial_corr(data=dat,x='x3',y='x2',covar='x1'))#p<0.01
x=np.column_stack((x1,x2,x3))
x=sm.add_constant(x)
model=sm.OLS(y,x)
result=model.fit()
print(result.summary())
yfit=result.fittedvalues
print(result.params)
# 模型的拟合优度可决系数
print(sum((yfit-np.mean(y))**2)/sum((y-np.mean(y))**2))

#计算x与y的决定系数
print(f"x与y的决定系数为{1-(sum((yfit-y)**2)/(x.shape[0]-1-1))/(sum((y-np.mean(y))**2)/(x.shape[0]-1))}")
print(1-(sum((yfit-y)**2)/(x.shape[0]-1-1))/(sum((y-np.mean(y))**2)/(x.shape[0]-1)))
# 回归方程的显著性检验
import scipy
ssr=sum((yfit-np.mean(y))**2)
sse=sum((yfit-y)**2)
print((ssr/3)/(sse/(10-3-1)))
f_val=((ssr/3)/(sse/(10-3-1)))
f_pval=stats.f.sf(f_val,3,6)
print(f_pval)
#回归系数显著性检验
S = np.array(np.linalg.inv(np.dot(np.mat(x).T,x)))
for i,col in enumerate(dat.columns):
    tval=result.params[i]/np.sqrt((sse/10-3-1)*S[i][i])
    pval=stats.t.sf(np.abs(tval),df=10-3-1)*2
    print(tval,pval)
# In[]捕食者模型
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import ipywidgets as ipw
alpha=1
beta=1
delta=1
gamma=1
x0=4
y0=2
def derivative(X,t,alpha,beta,delta,gamma):
    x,y=X
    dotx=x*(alpha-beta*y)#deer
    doty=y*(-delta+gamma*x)#wolve
    return np.array([dotx,doty])
nt=1000
tmax=30
t=np.linspace(0,tmax,nt)
x0=[x0,y0]
res=integrate.odeint(derivative,x0,t,args=(alpha,beta,delta,gamma))
x,y=res.T
plt.figure(dpi=600)
plt.grid()
plt.title("odeint method")
plt.plot(t,x,'xb',label='deer')
plt.plot(t,y,'+r',label='wolve')
plt.xlabel("Times t,[days]")
plt.ylabel("population")
plt.legend()
plt.show()

# In[]
plt.figure()
IC=np.linspace(1.0,6.0,21)
count=0
for deer in IC:
    x0=[deer,1.0]
    xs=integrate.odeint(derivative,x0,t,args=(alpha,beta,delta,gamma))
    if count %3==0:
        plt.plot(xs[:,0],xs[:,1],'-',label="$x_0=$"+str(x0[0]))
    count+=1
plt.xlabel("deer")
plt.ylabel("wolve")
plt.legend()
plt.title("deer vs wolve")
plt.show()

# In[]leslie模型
from scipy.linalg import leslie
print(leslie([0.1,2.0,1.0,0.1], [0.2,0.8,0.7]))
# In[]大象模型
#大象每3.5年怀一胎,0.0135概率双胞胎
born=(1/3.5)+(2/3.5)*0.0135#平均每年产几胎
x1=np.array([103,77,71,70,68,61,58,51,51,50,51,48,47,49,48,47,43,42,42,
             37,39,41,42,43,45,48,49,47,46,43,44,44,46,49,47,48,46,41,
             41,42,43,38,34,34,33,30,35,26,21,18,14,5,9,7,6,0,4,4,4,3,
             2,2,1,3,0,2,1,0,2,1])
x2=np.array([98,74,69,61,60,54,52,59,58,57,60,63,64,60,63,59,52,55,49,50,
             53,57,65,53,56,50,53,49,43,40,38,35,37,33,20,33,30,29,29,26,
             10,24,25,22,21,22,11,21,19,15,5,10,9,7,6,5,4,7,0,2,3,0,2,0,
             2,0,1,0,0,1])

y1=np.array([50,36,41,29,31,30,28,24,22,29,27,27,26,27,26,25,28,27,19,25,
             18,16,19,24,17,25,21,26,29,27,24,22,20,22,24,24,23,25,21,24,
             24,19,26,20,20,15,16,13,20,11,10,9,8,4,4,4,3,0,3,2,0,2,1,1,1,
             0,3,0,0,1])
y2=np.array([57,34,33,29,34,28,27,31,35,35,26,36,38,30,33,34,24,30,21,30,29,
             27,40,23,29,24,21,26,24,16,17,16,18,18,15,18,12,17,16,13,6,11,14,
             10,10,12,8,11,12,9,6,4,5,4,4,2,3,2,4,0,2,1,0,0,0,1,0,1,0,0])
# 1-61岁小象存活率
tt=x1[0:62]+x2[0:62]
tt1=tt[0:61]
tt2=tt[1:62]
tn=tt2/tt1
print(tn)
print(np.mean(tn))
# 求初始生存率
print((sum(x2)/sum(x1)))
# 求每年龄段生存率
n1=np.array([0]*70,dtype='float32')
n1[0]=1
n1[1]=(sum(x2)/sum(x1))
for i in range(2,61):
    n1[i]=((sum(x2)/sum(x1))+np.random.randn()*0.01)
for i in range(61,70):
    n1[i]=n1[i-1]*0.951
# 生存率和生存占比绘图
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

x=x1+x2
lv=(x[1:-1]/x[:-2])
t1=np.linspace(0,70,70)
t2=np.linspace(1,69,68)
plt.figure(dpi=1000)
plt.plot(t1,n1,"*g",linewidth=2,label="存活率")
plt.plot(t2,lv,"xr",linewidth=2,label="生存占比")
plt.grid(True)
plt.legend()
plt.xlabel("大象年龄")
plt.show()
#象群年龄结构
yy=100*x/sum(x)
a=np.arange(0,70)
plt.figure(dpi=1000)
plt.bar(a,yy,color='saddlebrown',width=2)
plt.xlabel("年龄")
plt.ylabel("占比")
plt.title("象群0-69年龄结构占比")
plt.show()
# In[]莱斯利预测下一年样本
n_next=n1[0:-1]
born=np.zeros([1,70])
s0=1/3.5+(2/3.5)*0.0135
#假设大象生育率线性递减
born[0,10:60]=s0
for i in range(60,70):
    born[0,i]=born[0,i-1]*(70-i)/60
from scipy.linalg import leslie
born=born.tolist()
born=sum(born,[])
n_next=n_next.tolist()
leslie_matrix=leslie(born,n_next)
print(leslie(born,n_next))
leslie_matrix=np.array(leslie_matrix)
# In[]
y_now=np.array((y2)).reshape(-1,1)
print(sum(y_now))
y_next_female=leslie_matrix.dot(y_now)
print(sum(y_next_female))
# In[]
for i in range(10):
    yyy=y_next_female
    yyy=np.array(yyy).reshape(-1,1)
    y_next_female=leslie_matrix.dot(yyy)
    print(sum(y_next_female))
# In[]插值方法对比方法
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
x0=[1,2,3,4,5];
y0=[1.6,1.8,3.2,5.9,6.8]
x=np.arange(1,5,1/30)
f1=interp1d(x0, y0,'linear')
y1=f1(x)
f2=interp1d(x0, y0,'zero')
y2=f2(x)
f3=interp1d(x0, y0,'slinear')
y3=f3(x)
f4=interp1d(x0, y0,'quadratic')
y4=f4(x)
f5=interp1d(x0, y0,'cubic')
y5=f5(x)
f6=interp1d(x0, y0,'nearest')
y6=f6(x)
f7=interp1d(x0, y0,'nearest-up')
y7=f7(x)
def lagrange(x0,y0,x):
    y=[]
    for k in range(len(x)):
        s=0
        for i in range(len(y0)):
            t=y0[i]
            for j in range(len(y0)):
                if i!=j:
                    t*=(x[k]-x0[j])/(x0[i]-x0[j])
            s+=t
        y.append(s)
    return y
y8=lagrange(x0,y0,x)
plt.figure(dpi=600)
plt.plot(x0,y0,'r*')
plt.plot(x,y1,'-',x,y2,'-',x,y3,'-',x,y4,'-',x,y5,'-',x,y6,'-',x,y7,'-',x,y8,'-')
plt.legend(["point","linear",'zero','slinear','quadratic',"cubic",'nearest','nearest-up',"lagrange",])
plt.grid()
plt.show()
# In[]
from scipy import stats
print(stats.ttest_1samp(y0,np.mean(y0)))
print(stats.ttest_1samp(y1,np.mean(y0)))
print(stats.ttest_1samp(y2,np.mean(y0)))#zero插值出现问题
print(stats.ttest_1samp(y3,np.mean(y0)))
print(stats.ttest_1samp(y4,np.mean(y0)))
print(stats.ttest_1samp(y5,np.mean(y0)))
print(stats.ttest_1samp(y6,np.mean(y0)))
print(stats.ttest_1samp(y7,np.mean(y0)))
# In[]
import numpy as np
import geatpy as ea
class myproblem(ea.Problem):
    def __init__(self):
        name='bnh'
        M = 3 # 初始化M（目标维数）
        maxormins = [1]*M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 4  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [65,1,0.55,np.pi*5/18]  # 决策变量下界
        ub = [100,5,0.94,np.pi*8/18]  # 决策变量上界
        lbin = [1, 1, 1,1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    def evalVars(self, Vars):  # 目标函数
        l1 = Vars[:, [0]]
        h = Vars[:, [1]]
        po = Vars[:, [2]]
        ceta = Vars[:, [3]]
        f1=(l1*np.cos(ceta))*40
        f2=np.sqrt((po*l1*np.cos(ceta)-40)**2+(po*l1*np.cos(ceta))**2)-(po*l1-40)
        f3=l1*40*h
        # 采用可行性法则处理约束
        objv=np.hstack([f1,f2,f3])
        CV = np.hstack([l1*np.sin(ceta)+h==70,
                        po>(40/l1),
                        (40+((po*l1*np.cos(ceta)-40)*(l1-40)))/np.sqrt((po*l1*np.cos(ceta)-40)**2+(po*l1*np.sin(ceta))**2)>0,
                        np.sqrt((po*l1*np.cos(ceta)-40)**2+(po*l1*np.sin(ceta))**2)<l1-40,
                        l1*np.cos(ceta)>40,
                        l1*np.sin(ceta)<=70,
                        ])
        return objv,CV
if __name__ == '__main__':
    # 实例化问题对象
    problem = myproblem()
    # 构建算法
    algorithm = ea.moea_NSGA3_templet(problem,
                                              ea.Population(Encoding='RI', NIND=100),
                                              MAXGEN=2000,  # 最大进化代数。
                                              logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.F = 0.95  # 差分进化中的参数F
    algorithm.recOper.XOVR = 0.95  # 重组概率
    # 求解
    res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
    print(res)

# In[]遗传算法
import numpy as np
def schaffer(p):
    x1,x2=p
    x=np.square(x1)+np.square(x2)
    return 0.5+(np.square(np.sin(x))-0.5)/np.square(1+0.001*x)
from sko.GA import GA
ga=GA(func=schaffer,n_dim=2,size_pop=50,max_iter=800,prob_mut=0.001,lb=[-1,-1],ub=[1,1],precision=1e-7)
best=ga.run()
print("best_x:",best[0][0],"best_y:",best[0][1])


# In[]
import pandas as pd
import matplotlib.pyplot as plt
y_history=pd.DataFrame(ga.all_history_Y)
plt.plot(y_history.index,y_history.values,'.',color='red')
y_history.min(axis=1).cummin().plot(kind='line')
plt.show()


# In[]
from scipy import spatial
from sko.GA import GA_TSP
num_points=50
points_coordinate=np.random.rand(num_points,2)
distance_matrix=spatial.distance.cdist(points_coordinate,points_coordinate,metric='euclidean')

def cal_total_distance(routine):
    num_points,=routine.shape
    return sum(distance_matrix[routine[i%num_points],routine[(i+1)%num_points]] for i in range(num_points))


ga_tsp=GA_TSP(func=cal_total_distance, n_dim=num_points,size_pop=50,max_iter=500,prob_mut=1)
best_points,best_distance=ga_tsp.run()

fig,ax=plt.subplots(1,2)
best_points_=np.concatenate([best_points,[best_points[0]]])
best_points_coordinate=points_coordinate[best_points_,:]
ax[0].plot(best_points_coordinate[:,0],best_points_coordinate[:,1],'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()
# In[]
def demo_func(x):
    x1,x2,x3=x
    return x1**2+(x2-0.05)**2+x3**2
from sko.PSO import PSO
pso=PSO(func=demo_func,dim=3,pop=40,max_iter=150,lb=[0,-1,0.5],ub=[1,1,1],w=0.8,c1=0.5,c2=0.5)
pso.run()
print("best_x:",pso.gbest_x,"best_y:",pso.gbest_y)

import matplotlib.pyplot as plt
plt.plot(pso.gbest_y_hist)
plt.show()

# In[]
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
wine=load_wine()
Xtrain,Xtest,Ytrain,Ytest=train_test_split(wine.data,wine.target,test_size=0.3)
clf=DecisionTreeClassifier()
clf.fit(Xtrain,Ytrain)
Ypredict=clf.predict(Xtest)
print(classification_report(Ypredict,Ytest))
# In[]
from sklearn import tree
from matplotlib import pyplot as plt
# 绘制图像
plt.figure(dpi=600)
_ = tree.plot_tree(clf,filled = True,feature_names=wine.feature_names) # 由于返回值不重要，因此直接用下划线接收
plt.show()
# In[]
import numpy as np
import math
import random
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
mpl.rcParams['axes.unicode_minus'] = False
n=900#换电需求数
min_price=170 #换电价格范围
max_price=230
A=np.random.normal(36,5,25)
# In[]
E=np.floor(A)
a=sum(E)-n#误差
A=A-a/25
E=np.floor(A)
b=sum(E)-n#修改
A=A-b/25
E=np.floor(A)#期望的
# In[]
a1=0.05;a2=0.95#距离成本与换电价格权重
x=[random.random()*20000 for i in range(n)]
y=[random.random()*20000 for i in range(n)]
H=np.mat([[2,2],[2,6],[2,10],[2,14],[2,18],
          [6,2],[6,6],[6,10],[6,14],[6,18],
          [10,2],[10,6],[10,10],[10,14],[10,18],
          [14,2],[14,6],[14,10],[14,14],[14,18],
          [18,2],[18,6],[18,10],[18,14],[18,18]])*1000
plt.plot(x,y,'r*')
plt.plot(H[:,0],H[:,1],'bo')
plt.legend(['司机','换电站'],loc='upper right',scatterpoints=1)
plt.title("初始位置图")
plt.show()
# In[]
#计算期望和实际期望
D=np.zeros((len(H),n))
price=200*np.ones((1,25))
for i in range(len(H)):
    for j in range(len(x)):
        D[i,j]=a1*np.sqrt(((H[i,0]-x[j]))**2+(H[i,1]-y[j])**2)+a2*price[0,i]
D=D.T
D=D.tolist()
d2=[D[i].index(np.min(D[i])) for i in range(n)]#最小dij下标
C=Counter(d2)#记录每个换电站存放有多少汽车
e=list(C.values())
err=sum(abs(E-e))
# In[]
#博弈过程
J=[]#价格变化的差值
ER=[err]#E-e的变化差值
for k in range(1,100):
    j=0
    for i in range(25):
        if e[i]<E[i] and price[0,i]>=min_price:
            price[0,i]=price[0,i]-1
            j=j+1
        if e[i]>E[i] and price[0,i]<=max_price:
            price[0,i]=price[0,i]+1
            j=j+1
    J.append(j)
    DD=np.zeros((len(H),n))#需求车辆到各换电站的需求比例
    for i in range(len(H)):
        for j in range(len(x)):
            DD[i,j]=a1*np.sqrt(((H[i,0]-x[j]))**2+(H[i,1]-y[j])**2)+a2*price[0,i]
    DD=DD.T
    DD=DD.tolist()
    dd2=[DD[i].index(np.min(DD[i])) for i in range(n)]
    C=Counter(dd2)
    e=[C[i] for i in sorted(C.keys())]
    err=sum(abs(E-e))
    ER.append(err)
plt.figure(dpi=500)
plt.plot(ER,'-o')
plt.title('E-e的差值变化')
plt.legend('E-e')
plt.grid()
plt.show()

plt.figure(dpi=500)
plt.bar(x=range(1,26),
        height=price[0],
        color='steelblue',
        width=0.8
        )
plt.plot([1,26],[min_price,min_price],'g--')
plt.plot([1,26],[max_price,max_price],'r--')
plt.title("换电站的换电价格")
plt.ylabel("price")
plt.axis([0,26,0,300])
plt.show()

plt.figure(dpi=500)
index=np.arange(1,26)
rects1=plt.bar(index,e,0.5,color="#0072BC")
rects2=plt.bar(index+0.5,E,0.5,color="#ED1C24")
plt.axis([0,26,0,50])
plt.title("出租车的预期和实际数量")
plt.ylabel("E and e")
plt.xlabel("换电站")
plt.legend(['e','E'])
plt.show()

# In[]
import numpy as np
mu, sigma = 0, 0.1      # 均值和标准差
s = np.random.normal(mu, sigma, 1000)
import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, density=True, color='b')
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.show()
# In[]
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def regress():
  # 获取数据
  lb = load_boston()
  # 分割数据集为训练集和测试集
  x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
  # 进行标准化处理，特征值和目标值都需要分别进行特征化处理
  # 特征值进行标准化
  std_x = StandardScaler()
  x_train = std_x.fit_transform(x_train)
  x_test = std_x.transform(x_test)
  # 目标值进行标准化
  std_y = StandardScaler()
  y_train = std_y.fit_transform(y_train.reshape(-1,1))
  y_test = std_y.transform(y_test.reshape(-1,1))
  # estimator预测
  #正规方程求解方式预测结果
  lr = LinearRegression()
  lr.fit(x_train, y_train)
  print("特征值：",lr.coef_)
  # 预测测试集的房子价格
  y_predict = std_y.inverse_transform(lr.predict(x_test))
  print("正规方程求解每个房子的预测价格:",y_predict)

  #梯度下降求解预测结果
  SGD=SGDRegressor()
  SGD.fit(x_train, y_train)
  print("特征值",SGD.coef_)
  # 预测测试集的房子价格
  SGD_y_predict = std_y.inverse_transform(SGD.predict(x_test))
  print("梯度下降每个房子的预测价格:", SGD_y_predict)



  # 岭回归降求解预测结果
  ri = Ridge(alpha=1.0)
  ri.fit(x_train, y_train)
  print("特征值", ri.coef_)
  # 预测测试集的房子价格
  ri_y_predict = std_y.inverse_transform(ri.predict(x_test))
  print("梯度下降每个房子的预测价格:", ri_y_predict)
  print("正规方程均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict))
  print("梯度下降均方误差：", mean_squared_error(std_y.inverse_transform(y_test), SGD_y_predict))
  print("岭回归误差：", mean_squared_error(std_y.inverse_transform(y_test), ri_y_predict))
  return None
if __name__ == "__main__":
  regress()

