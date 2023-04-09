# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:32:21 2022

@author: zklei
"""
# In[]解方程
import numpy as np
a=np.array([[10,-1,-2],[-1,10,-2],[-1,-1,5]])#A为系数矩阵
b=np.array([72,83,42])#B为常数项
inv_a=np.linalg.inv(a)#求a的逆矩阵
# In[]
x=inv_a.dot(b)
x=np.linalg.solve(a,b)
print(x)
# In[]
from sympy import symbols,Eq,solve
x,y,z=symbols('x y z')
eqs=[Eq(10*x-y-2*z,72),Eq(-1*x+10*y-2*z,83),Eq(-x-y+5*z,42)]
print(solve(eqs,[x,y,z]))

# In[]
from scipy import optimize
c=np.array([2,3,-5])
A=np.array([[-2,5,-1],[1,3,1]])
b=np.array([-10,12])
Aeq=np.array([[1,1,1]])
beq=np.array([7])
x1=(0,None)
x2=(0,None)
x3=(0,None)
res=optimize.linprog(-c,A,b,Aeq,beq,bounds=(x1,x2,x3))
print(res)

# In[]
c=np.array([2,3,-1])
A=np.array([[1,1,3],[0,-2,1],[3,-1,-4]])
b=np.array([100,-15,-40])
Aeq=np.array([[1,1,1]])
beq=np.array([70])
x1=(0,None)
x2=(0,None)
x3=(0,None)
res=optimize.linprog(-c,A,b,Aeq,beq,bounds=(x1,x2,x3))
print(res)
# In[]
from scipy.optimize import minimize
import numpy as np
def func(x):
    return 10.5+0.3*x[0]+0.32*x[1]+0.32*x[2]+0.0007*x[0]**2+0.0004*x[1]**2+0.00045*x[2]**2
cons=({'type':'eq','func':lambda x:x[0]+x[1]+x[2]-700})
b1,b2,b3=(100,200),(120,250),(150,300)
x0=np.array([100,200,400])
res=minimize(func, x0,method='Nelder-Mead',bounds=(b1,b2,b3),constraints=cons)
print(res)
res=minimize(func, x0,method='Powell',bounds=(b1,b2,b3),constraints=cons)
print(res)
# In[]遗传算法
from sko.GA import GA
def func(x):
    return 10.5+0.3*x[0]+0.32*x[1]+0.32*x[2]+0.0007*x[0]**2+0.0004*x[1]**2+0.00045*x[2]**2
cons=[lambda x:x[0]+x[1]+x[2]-700,
      lambda x:x[0]+x[1]<x[2]]
b1,b2,b3=(100,200),(120,250),(150,300)
ga=GA(func=func,n_dim=3,size_pop=500,max_iter=200,constraint_eq=cons,lb=[100,120,150],ub=[200,250,300])
best_x,best_y=ga.run()
print("best_x:\n",best_x,"best_y:\n",best_y)
import pandas as pd
import matplotlib.pyplot as plt

Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()

# In[]
import numpy as np
import geatpy as ea

"""
    min f = 10.5+0.3*x1+0.32*x2+0.32*x3+0.0007*x1**2+0.0004*x2**2+0.00045*x3**2
    s.t.
    x1 + x2 + x3 - 700 == 0
    100 <= x1 <= 200
    120 <= x2 <= 250
    150 <= x3 <= 300
"""

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 3  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [100, 120, 150]  # 决策变量下界
        ub = [200, 250, 300]  # 决策变量上界
        lbin = [1, 1, 1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # 目标函数
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        f = 10.5+0.3*x1+0.32*x2+0.32*x3+0.0007*x1**2+0.0004*x2**2+0.00045*x3**2
        # 采用可行性法则处理约束
        CV=np.hstack([np.abs(x1+x2+x3-700)])
        return f, CV

    
"""
    该案例展示了一个带等式约束的连续型决策变量最大化目标的单目标优化问题的求解。
"""

if __name__ == '__main__':
    # 实例化问题对象
    problem = MyProblem()
    # 构建算法
    algorithm = ea.soea_DE_rand_1_bin_templet(problem,
                                              ea.Population(Encoding='RI', NIND=100),
                                              MAXGEN=200,  # 最大进化代数。
                                              logTras=50)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.F = 0.5  # 差分进化中的参数F
    algorithm.recOper.XOVR = 0.7  # 重组概率
    # 求解
    res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
    print(res)

# In[]
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
 
# 生成X和Y的数据
X = np.arange(100,200,0.1)  #生成-5，4.9之间间隔为0.1的一系列数据,100个数据
print(len(X))
Y = np.arange(120, 250, 0.1)
X, Y = np.meshgrid(X, Y)   #对应两个数组中所有的(x,y)对
a = 10
# 目标函数
Z = 10.5+0.3*X+0.32*Y+0.0007*X**2+0.0004*Y**2
  
# 绘图
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
plt.show()
# In[]案例
a=np.array([1.25,8.75,0.5,5.75,3,7.25])
b=np.array([1.25,0.75,4.75,5,6.5,7.25])
d=np.array([3,5,4,7,6,11])
def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)
def func(x):
    ssum=0
    a=np.array([1.25,8.75,0.5,5.75,3,7.25])
    b=np.array([1.25,0.75,4.75,5,6.5,7.25])
    for i in range(12):
            if i<=5:
                ssum=ssum+x[i]*np.sqrt((x[12]-a[i])**2+(x[13]-b[i])**2)
            else:
                ssum=ssum+x[i]*np.sqrt((x[14]-a[i-6])**2+(x[15]-b[i-6])**2)
    return -ssum
from scipy.optimize import minimize
import numpy as np
cons=({'type':'ineq','func':lambda x:20-(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]*0+x[8]*0+x[9]*0+x[10]*0+x[11]*0+x[12]*0+x[13]*0+x[14]*0+x[15]*0)},
      {'type':'ineq','func':lambda x:20-(x[0]*0+x[1]*0+x[2]*0+x[3]*0+x[4]*0+x[5]*0+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12]*0+x[13]*0+x[14]*0+x[15]*0)},
      {'type':'eq','func':lambda x: x[0]+x[1]*0+x[2]*0+x[3]*0+x[4]*0+x[5]*0+x[6]+x[7]*0+x[8]*0+x[9]*0+x[10]*0+x[11]*0+x[12]*0+x[13]*0+x[14]*0+x[15]*0-3},
      {'type':'eq','func':lambda x: x[0]*0+x[1]*1+x[2]*0+x[3]*0+x[4]*0+x[5]*0+x[6]*0+x[7]*1+x[8]*0+x[9]*0+x[10]*0+x[11]*0+x[12]*0+x[13]*0+x[14]*0+x[15]*0-5},
      {'type':'eq','func':lambda x: x[0]*0+x[1]*0+x[2]*1+x[3]*0+x[4]*0+x[5]*0+x[6]+x[7]*0+x[8]*1+x[9]*0+x[10]*0+x[11]*0+x[12]*0+x[13]*0+x[14]*0+x[15]*0-4},
      {'type':'eq','func':lambda x: x[0]*0+x[1]*0+x[2]*0+x[3]+x[4]*0+x[5]*0+x[6]+x[7]*0+x[8]*0+x[9]+x[10]*0+x[11]*0+x[12]*0+x[13]*0+x[14]*0+x[15]*0-7},
      {'type':'eq','func':lambda x: x[0]+x[1]*0+x[2]*0+x[3]*0+x[4]+x[5]*0+x[6]+x[7]*0+x[8]*0+x[9]*0+x[10]+x[11]*0+x[12]*0+x[13]*0+x[14]*0+x[15]*0-6},
      {'type':'eq','func':lambda x: x[0]+x[1]*0+x[2]*0+x[3]*0+x[4]*0+x[5]+x[6]+x[7]*0+x[8]*0+x[9]*0+x[10]*0+x[11]+x[12]*0+x[13]*0+x[14]*0+x[15]*0-11})
x0=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
bnd=[[0,d[0]],[0,d[1]],[0,d[2]],[0,d[3]],[0,d[4]],[0,d[5]],[0,d[0]],[0,d[1]],[0,d[2]],[0,d[3]],[0,d[4]],[0,d[5]],[0,None],[0,None],[0,None],[0,None]]
res=minimize(func,x0,method='Nelder-Mead',bounds=bnd,constraints=cons)
print(res)
# In[]任务指派
from scipy.optimize import linear_sum_assignment
import numpy as np
cost=np.array([[25,29,31,42],[39,38,26,20],[34,27,28,40],[24,42,36,23]])
row_ind,col_ind=linear_sum_assignment(cost)
print(row_ind)
print(col_ind)
print(cost[row_ind,col_ind])
print(cost[row_ind,col_ind].sum())
# In[]
from scipy.optimize import linear_sum_assignment
fight=np.array([[22,16,20,35,18],[20,12,35,40,26]
                ,[11,19,15,17,21],[25,30,21,37,40]
                ,[22,26,35,30,19]])
row_ind,col_ind=linear_sum_assignment(fight)
print(row_ind,col_ind)
print(fight[row_ind,col_ind])
print(fight[col_ind,col_ind].sum())
# In[]
a2=np.array([[4,9,64,169,225],[361,400,625,36,64],[225,256,441,4,16],[484,529,16,81,121],[196,225,400,625,9]])
b2=np.array([256,225,100,64,256]).reshape(-1,1).dot(np.ones([1,5]))
c2=np.array([[49,225,225,49],[25,169,169,25],[169,441,441,169],[64,256,256,64]])
row_ind,col_ind=linear_sum_assignment(a2)
print(row_ind,col_ind)
print(a2[row_ind,col_ind])
print(a2[col_ind,col_ind].sum())
a1=np.zeros([5,5])
a1[row_ind,col_ind]=1
print(a1)
row_ind,col_ind=linear_sum_assignment(b2)
print(row_ind,col_ind)
print(b2[row_ind,col_ind])
print(b2[col_ind,col_ind].sum())
b1=np.zeros([5,5])
b1[row_ind,col_ind]=1
print(b1)
row_ind,col_ind=linear_sum_assignment(c2)
print(row_ind,col_ind)
print(c2[row_ind,col_ind])
print(c2[col_ind,col_ind].sum())
c1=np.zeros([4,4])
c1[row_ind,col_ind]=1
print(c1)
# In[]
from scipy.integrate import quad
def integrand(x,a,b):
    return a*x**2+b
a=2
b=1
I=quad(integrand,0,1,args=(a,b))
print(I)
# In[]
import numpy as np
def integrand2(t,n,x):
    return np.exp(-x*t)/t**n
def expint(n,x):
    return quad(integrand2,1,np.inf,args=(n,x))[0]
vec_expint=np.vectorize(expint)
print(vec_expint(3, np.arange(1.0, 4.0, 0.5)))

# In[]
import scipy.special as special
print(special.expn(3, np.arange(1.0,4.0,0.5)))
result=quad(lambda x:expint(3,x),0,np.inf)
print(result)
# In[]
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np
x1=np.array([19,45,35,31,25,32,21,26,24,27,9,23,33,29])
ytrue=np.array([60,113,94,90,60,88,59,61,57,78,27,72,85,63])
X=sm.add_constant(x1)
model=sm.OLS(ytrue,X)
result=model.fit()
print(result.summary())
yfit=result.fittedvalues
print(result.params)
print("截距为:b=",result.params[0],"系数为:a=",result.params[1])
y_40_predict=result.params[0]+40*result.params[1]
print("预测值为:",y_40_predict)
prstd,ivlow,ivup=wls_prediction_std(result)#返回标准偏差和置信区间
from matplotlib import pyplot as plt
fig,ax=plt.subplots(figsize=(10,8))
ax.plot(x1,ytrue,'o',label="data")
ax.plot(x1,yfit,'r-',label="OLS")
ax.plot(x1,ivup,'--',color='orange',label='upconf')
ax.plot(x1,ivlow,'--',color='orange',label='lowconf')
ax.legend(loc='best')#显示图例
plt.title('OLS linear regression for Event Promotion')
plt.grid()
plt.show()
# In[]
import numpy as np
import geatpy as ea
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self,x,y):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 8  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [-100000]*Dim  # 决策变量下界
        ub = [100000]*Dim  # 决策变量上界
        lbin = [1]*Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # 目标函数
        a = Vars[:, [0]]
        c = Vars[:, [1]]
        x1 = Vars[:, [2]]
        y1 = Vars[:, [3]]
        z1 = Vars[:, [4]]
        x2 = Vars[:, [5]]
        y2 = Vars[:, [6]]
        z2 = Vars[:, [7]]
        f = np.abs(0.466*30-4/a-c)
        # 采用可行性法则处理约束
        CV=np.hstack([a*(x1**2+y1**2)+30+c+259,
                      -np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)+2-2*0.0007,
                      np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)-2-2*0.0007,
                      -np.sqrt((x1-1)**2+(y1-1)**2+(z1-1)**2)-0.6,
                      np.sqrt((x1-1)**2+(y1-1)**2+(z1-1)**2)-0.6,
                      np.sqrt((x2-2)**2+(y2-2)**2+(z2-2)**2)-0.6,
                      -np.sqrt((x2-1)**2+(y2-1)**2+(z2-1)**2)-0.6
                      ])
        return f, CV

    
"""
    该案例展示了一个带等式约束的连续型决策变量最大化目标的单目标优化问题的求解。
"""

if __name__ == '__main__':
    # 实例化问题对象
    problem = MyProblem(1,1)
    # 构建算法
    algorithm = ea.soea_DE_rand_1_bin_templet(problem,
                                              ea.Population(Encoding='RI', NIND=100),
                                              MAXGEN=1000,  # 最大进化代数。
                                              logTras=50)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.F = 0.5  # 差分进化中的参数F
    algorithm.recOper.XOVR = 0.7  # 重组概率
    # 求解
    res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
    print(res)

# In[]
#选择问题，基站最小数量+最坏到达时间最短
import numpy as np
import geatpy as ea
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [1]*M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 8+64+1  # 初始化Dim（决策变量维数）
        varTypes = [1] * 72+[1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0]*72+[0]  # 决策变量下界
        ub = [1]*72+[10]  # 决策变量上界
        lbin = [1]*72+[0]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        self.r=np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 0, 0, 0, 0],
                         [0, 1, 1, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 1, 1]])
        self.t=np.array([[7, 0, 0, 0, 0, 0, 0, 0],
                         [0, 5, 8, 0, 0, 0, 0, 0],
                         [0, 9, 4, 0, 10, 0, 0, 0],
                         [0, 0, 0, 10, 0, 0, 0, 0],
                         [0, 0, 0, 0, 9, 0, 0, 0],
                         [0, 0, 0, 0, 0, 6, 10, 0],
                         [0, 0, 0, 0, 0, 0, 5, 9],
                         [0, 0, 0, 0, 0, 0, 8, 6]])
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, Vars):  # 目标函数
        f1=0
        for i in range(8):
            f1+=Vars[:,[i]]
        cv=np.zeros((100,88))
        f2=Vars[:,[72]]
        for i in range(8):
            temp=0
            for j in range(8):
                temp+=Vars[:,[j]]*self.r[i,j]
            cv[:,[i]]=1-temp
        for i in range(8,72):
            cv[:,[i]]=Vars[:,[i]]-Vars[:,[int(i/8)-1]]
        for i in range(72,80):
            temp=0
            for j in range(8):
                temp+=Vars[:,(i-71)*8+j]
            cv[:,i]=temp-Vars[:,i-72]
        for i in range(80,88):
            temp=0
            for j in range(8):
                temp+=Vars[:,8*(i-79)+j]*Vars[:,j]*self.t[i-80,j]
            cv[:,i]=temp-Vars[:,72]
        # 采用可行性法则处理约束
        return np.hstack([f1,f2]), cv
if __name__ == '__main__':
    # 实例化问题对象
    problem = MyProblem()
    # 构建算法
    algorithm = ea.moea_NSGA2_DE_templet(problem,
                                              ea.Population(Encoding='RI', NIND=100),
                                              MAXGEN=100,  # 最大进化代数。
                                              logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.F = 0.95  # 差分进化中的参数F
    algorithm.recOper.XOVR = 0.95  # 重组概率
    # 求解
    res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=False)
    print(res)
# In[]


