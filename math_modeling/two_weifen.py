# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:55:53 2022

@author: zklei
"""
# In[]
def dynamic_p()->list:
    items=[
        {"name":"水","weight":3,"value":10},
        {"name":"书","weight":1,"value":3},
        {"name":"食物","weight":2,"value":9},
        {"name":"小刀","weight":3,"value":4},
        {"name":"衣物","weight":2,"value":5},
        {"name":"手机","weight":1,"value":10},
        ]
    max_capacity=6
    dp=[[0]*(max_capacity+1)for _ in range(len(items)+1)]
    for row in range(1,len(items)+1):
        for col in range(1,max_capacity+1):
            weight=items[row-1]["weight"]
            value=items[row-1]["value"]
            if weight>col:
                dp[row][col]=dp[row-1][col]
            else:
                dp[row][col]=max(dp[row-1][col-weight]+value,dp[row-1][col])
    return dp,items
dp,items=dynamic_p()
print(dp[6][6])

for i in dp:
    print(i)
x=[0]*7
j=6
i=6
while i>=0:
    if dp[i][j]>dp[i-1][j]:
        x[i]=1
        j-=items[i-1]["weight"]
    else:
        x[i]=0
    i-=1
for i in range(len(x)):
    if x[i]==1:
        print(i,end=" ")
# In[]解析解
from sympy import *
y=symbols('y',cls=Function)
x=symbols('x')
eq=Eq(y(x).diff(x,2)+4*y(x).diff(x,1)+29*y(x),0)
print(dsolve(eq,y(x)))
c1=symbols('c1')
c2=symbols('c2')
f=(c1*sin(5*x)+c2*cos(5*x))*exp(-2*x)
print(f.diff(x,1))
# In[]
import sympy as sy
x=sy.symbols("x")
omega=sy.symbols("w")
f=sy.Function("f")
equation=f(x).diff(x,2)+omega**2*f(x)
print(sy.dsolve(equation,f(x)))
#练习
x=sy.symbols("x")
f=sy.Function("f")
equation=f(x).diff(x,4)-2*f(x).diff(x,3)+5*f(x).diff(x,2)
print(sy.dsolve(equation,f(x)))
# In[]
from math import e
equation=f(x).diff(x,2)-5*f(x).diff(x,1)+6*f(x)-x*e**(2*x)
print(sy.dsolve(equation,f(x)))
# In[]
import matplotlib.pyplot as plt
import scipy.integrate as sp
import numpy as np
b=0.25
c=5.0
def function(t,y):#第一个为自变量t,第二个为相对
    dy1=y[1]
    dy2=-b*dy1-c*np.sin(y[0])
    return [dy1,dy2]
t=np.linspace(0,10,100)
init_x=[np.pi-0.1,0]
result=sp.odeint(function,init_x,t,tfirst=True)
print(result)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

plt.xlabel('t')
plt.ylabel('f(t)')
plt.plot(t, result[:, 0], label='f:yt')#原函数数值解
plt.plot(t, result[:, 1], label='f:zt')#一阶导
plt.legend()
plt.show()
# In[]
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import os
#先从odeint函数直接求解微分方程
#创建欧拉法的类
class Euler:
    #构造方法，当创建对象的时候，自动执行的函数
    def __init__(self,h,y0):
        #将对象与对象的属性绑在一起
        self.h = h
        self.y0 = y0
        self.y = y0
        self.n = 1/self.h
        self.x = 0
        self.list = [1]
        #欧拉法用list列表,其x用y叠加储存
        self.list2 = [1]
        self.y1 = y0
        #改进欧拉法用list2列表，其x用y1叠加储存
        self.list3 = [1]
        self.y2 = y0
        #隐式梯形法用list3列表，其x用y2叠加储存
    #欧拉法的算法，算法返回t,x
    def countall(self):
        for i in range(int(self.n)):
            y_dere = -20*self.list[i]
            #欧拉法叠加量y_dere = -20 * x
            y_dere2 = -20*self.list2[i] + 0.5*400*self.h*self.list2[i]
            #改进欧拉法叠加量 y_dere2 = -20*x(k) + 0.5*400*delta_t*x(k)
            y_dere3 = (1-10*self.h)*self.list3[i]/(1+10*self.h)
            #隐式梯形法计算 y_dere3 = (1-10*delta_t)*x(k)/(1+10*delta_t)
            self.y += self.h*y_dere
            self.y1 += self.h*y_dere2
            self.y2 =y_dere3
            self.list.append(float("%.10f" %self.y))
            self.list2.append(float("%.10f"%self.y1))
            self.list3.append(float("%.10f"%self.y2))
        return np.linspace(0,1,int(self.n+1)), self.list,self.list2,self.list3

step = 0.001
step = float(step)
work1 = Euler(step,1)
ax1,ay1,ay2,ay3 = work1.countall()
#画图工具plt
plt.figure(1)
plt.subplot(1,3,1)
plt.plot(ax1,ay1,'s-.')
plt.xlabel('横坐标t',fontproperties = 'simHei',fontsize =20)
plt.ylabel('纵坐标x',fontproperties = 'simHei',fontsize =20)
plt.title('欧拉法求解微分线性方程步长为'+str(step),fontproperties = 'simHei',fontsize =20)

plt.subplot(1,3,2)
plt.plot(ax1,ay2,'s-.')
plt.xlabel('横坐标t',fontproperties = 'simHei',fontsize =20)
plt.ylabel('纵坐标x',fontproperties = 'simHei',fontsize =20)
plt.title('改进欧拉法求解微分线性方程步长为'+str(step),fontproperties = 'simHei',fontsize =20)

plt.subplot(1,3,3)
plt.plot(ax1,ay3,'s-.')
plt.xlabel('横坐标t',fontproperties = 'simHei',fontsize =20)
plt.ylabel('纵坐标x',fontproperties = 'simHei',fontsize =20)
plt.title('隐式梯形法求解微分线性方程步长为'+str(step),fontproperties = 'simHei',fontsize =20)

plt.figure(2)
plt.plot(ax1,ay1,ax1,ay2,ax1,ay3,'s-.')
plt.xlabel('横坐标t',fontproperties = 'simHei',fontsize =20)
plt.ylabel('纵坐标x',fontproperties = 'simHei',fontsize =20)
plt.title('三合一图像步长为'+str(step),fontproperties = 'simHei',fontsize =20)
ax = plt.gca()
ax.legend(('$Eular$','$fixed Eular$','$trapezoid$'),loc = 'lower right',title = 'legend')
plt.show()
os.system("pause")

# In[]练习
from sympy import *
y=symbols('y',cls=Function)
x=symbols('x')
eq=Eq(y(x).diff(x,2)-4*y(x).diff(x,1)+3*y(x)-x*exp(x))
print(dsolve(eq,y(x)))
# In[]
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np
def fvdp(x,y):
    dy1=y[1]#一阶导
    dy2=4*y[1]-3*y[0]+x*np.exp(x)
    return [dy1,dy2]
def sovle_second_order_ode():
    x=np.arange(0,0.25,0.01)
    y0=[0,1]
    y=odeint(fvdp,y0,x,tfirst=True)
    y1, =plt.plot(x,y[:,0],label='y')
    y1_1, =plt.plot(x,y[:,1],label='y_one_differ')
    plt.legend(handles=[y1,y1_1])
    plt.show()
    return y
y=sovle_second_order_ode()
# In[]
from matplotlib import pyplot as plt
def fvdp(t,y):
    dy1=y[1]
    dy2=1000*(1-y[0]**2)*y[1]-y[0]
    return [dy1,dy2]
def solve_second_order_ode():
    x=np.arange(0,0.25,0.01)
    y0=[0.0,2.0]
    y=odeint(fvdp,y0,x,tfirst=True)
    
    y1, =plt.plot(x,y[:,0],label='y')
    y1_1, =plt.plot(x,y[:,1],label='y_one_differ')
    plt.legend(handles=[y1,y1_1])
    plt.show()
    return y
y=solve_second_order_ode()


# In[]数值解
from scipy.integrate import odeint
import numpy as np
dy=lambda y,x:1/(1+x**2)-2*y**2
x=np.arange(0,10.5,0.5)
sol=odeint(dy,0,x)
print("x={}\n对应的数值y={}".format(x,sol.T))
from matplotlib import pyplot as plt
plt.plot(x,sol)
plt.grid()
plt.show()

# In[]
def fvdp(t,y):
    dy1=y[1]
    dy2=1000*(1-y[0]**2)*y[1]-y[0]
    return [dy1,dy2]
def solve_second_order_ode():
    x=np.arange(0,0.25,0.01)
    y0=[0.0,2.0]
    y=odeint(fvdp,y0,x,tfirst=True)
    
    y1, =plt.plot(x,y[:,0],label='y')
    y1_1, =plt.plot(x,y[:,1],label='y_one_differ')
    plt.legend(handles=[y1,y1_1])
    plt.show()
    return y
y=solve_second_order_ode()

# In[]微分方程组
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
from scipy.integrate import solve_ivp
def fun(t,w):
    x=w[0]
    y=w[1]
    z=w[2]
    return [2*x-3*y+3*z,4*x-5*y+3*z,4*x-4*y+2*z]
y0=[1,2,1]

yy=solve_ivp(fun,(0,10), y0,method='RK45',t_eval=np.arange(0,10,1))
t=yy.t
data=yy.y
plt.figure(dpi=600)
plt.plot(t,data[0,:],label='X')
plt.plot(t,data[1,:],label='Y')
plt.plot(t,data[2,:],label='Z')
plt.legend()
plt.xlabel("时间")
plt.show()
# In[]
import numpy as np
import matplotlib.pyplot as plt
rc=2#常数
dt=0.5#步长
n=1000#分割段数
t=0
q=1
#先定义三个空列表
qt=[] #用来盛放差分得到的q值
qt0=[] #用来盛放解析得到的q值
time = [] #用来盛放时间值
for i in range(n):
    t=t+dt
    q1 = q - q*dt/rc #qn+1的近似值
    q = q - 0.5*(q1*dt/rc + q*dt/rc) #差分递推关系
    q0 = np.exp(-t/rc) #解析关系
    qt.append(q) #差分得到的q值列表
    qt0.append(q0) #解析得到的q值列表
    time.append(t) #时间列表
plt.plot(time,qt,'o',label='Euler-Modify') #差分得到的电量随时间的变化
plt.plot(time,qt0,'r-',label='Analytical') #解析得到的电量随时间的变化
plt.xlabel('time')
plt.ylabel('charge')
plt.xlim(0,20)
plt.ylim(-0.2,1.0)
plt.legend(loc='upper right')
plt.show()
# In[]
from numpy import *
import matplotlib.pyplot as plt

h = 0.1#空间步长
N =30#空间步数
dt = 0.0001#时间步长
M = 10000#时间的步数
A = dt/(h**2) #lambda*tau/h^2
U = zeros([N+1,M+1])#建立二维空数组
Space = arange(0,(N+1)*h,h)#建立空间等差数列，从0到3，公差是h

#边界条件
for k in arange(0,M+1):
    U[0,k] = 0.0
    U[N,k] = 0.0

#初始条件
for i in arange(0,N):
    U[i,0]=4*i*h*(3-i*h)

#递推关系
for k in arange(0,M):
    for i in arange(1,N):
        U[i,k+1]=A*U[i+1,k]+(1-2*A)*U[i,k]+A*U[i-1,k]
#不同时刻的温度随空间坐标的变化
plt.plot(Space,U[:,0], 'g-', label='t=0',linewidth=1.0)
plt.plot(Space,U[:,3000], 'b-', label='t=3/10',linewidth=1.0)
plt.plot(Space,U[:,6000], 'k-', label='t=6/10',linewidth=1.0)
plt.plot(Space,U[:,9000], 'r-', label='t=9/10',linewidth=1.0)
plt.plot(Space,U[:,10000], 'y-', label='t=1',linewidth=1.0)
plt.ylabel('u(x,t)', fontsize=20)
plt.xlabel('x', fontsize=20)
plt.xlim(0,3)
plt.ylim(-2,10)
plt.legend(loc='upper right')
plt.show()

# In[]
def fun2(t,w):
    x=w[0]
    y=w[1]
    return [-x**3-y,-y**3+x]
y0=[1,0.5]
yy=solve_ivp(fun2,(0,100), y0,method='RK45',t_eval=np.arange(1,100,1))
t=yy.t
data=yy.y
plt.plot(t,data[0,:],label='X')
plt.plot(t,data[1,:],label='Y')
plt.legend()
plt.xlabel("时间")
plt.show()
# In[]
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import numpy as np
def fun3(w,t,a,b,c):
    x=w[0]
    y=w[1]
    z=w[2]
    return [a*(y-x),x*(b-z)-y,x*y-c*z]
y0=[1.0,2.0,1.0]
t=np.arange(0,30,0.02)
track1=odeint(fun3,y0,t,args=(10,28,3))
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.plot(track1[:,0],track1[:,1],track1[:,2],lw=1)

# In[]
from matplotlib import pyplot as plt
from scipy.integrate import odeint
def pfun(x,y):
    dy1=y[1]
    dy2=-2*y[0]-2*dy1
    #dy2+2*dy1+2*y[0]
    return [dy1,dy2]
x=np.arange(0,10,0.1)
soli=odeint(pfun,[0,1],x,tfirst=True)
plt.figure(dpi=600)
plt.rc('font',size=16); plt.rc('font',family='SimHei')
plt.plot(x,soli[:,0],'r*',label="数值解")
plt.plot(x,np.exp(-x)*np.sin(x),'g',label="符号解曲线")
plt.legend()
plt.show()

# In[]
# 1. 求解微分方程初值问题(scipy.integrate.odeint)
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np
import matplotlib.pyplot as plt

def dy_dt(y, t):  # 定义函数 f(y,t)
    return np.sin(t**2)

y0 = [1]  # y0 = 1 也可以
t = np.arange(-10,10,0.01)  # (start,stop,step)
y = odeint(dy_dt, y0, t)  # 求解微分方程初值问题

# 绘图
plt.plot(t, y)
plt.title("scipy.integrate.odeint")
plt.show()

# In[]
# 2. 求解微分方程组初值问题(scipy.integrate.odeint)
from scipy.integrate import odeint    # 导入 scipy.integrate 模块
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导数函数, 求 W=[x,y,z] 点的导数 dW/dt
def lorenz(W,t,p,r,b):  # by youcans
    x, y, z = W  # W=[x,y,z]
    dx_dt = p*(y-x)  # dx/dt = p*(y-x), p: sigma
    dy_dt = x*(r-z) - y  # dy/dt = x*(r-z)-y, r:rho
    dz_dt = x*y - b*z  # dz/dt = x*y - b*z, b;beta
    return np.array([dx_dt,dy_dt,dz_dt])

t = np.arange(0, 30, 0.01)  # 创建时间点 (start,stop,step)
paras = (10.0, 28.0, 3.0)  # 设置 Lorenz 方程中的参数 (p,r,b)

# 调用ode对lorenz进行求解, 用两个不同的初始值 W1、W2 分别求解
W1 = (0.0, 1.00, 0.0)  # 定义初值为 W1
track1 = odeint(lorenz, W1, t, args=(10.0, 28.0, 3.0))  # args 设置导数函数的参数
W2 = (0.0, 1.01, 0.0)  # 定义初值为 W2
track2 = odeint(lorenz, W2, t, args=paras)  # 通过 paras 传递导数函数的参数
# 绘图
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(track1[:,0], track1[:,1], track1[:,2], color='magenta') # 绘制轨迹 1
ax.plot(track2[:,0], track2[:,1], track2[:,2], color='deepskyblue') # 绘制轨迹 2
ax.set_title("Lorenz attractor by scipy.integrate.odeint")
plt.show()

# In[]
def fvdp(t,y,r,l,lc):
    dy1=y[1]
    dy2=-(y[0]/lc+r*y[1]/l)
    return [dy1,dy2]
def solve_second_order_ode():
    x=np.arange(0,20,0.01)
    y0=[1.0,0.0]
    y=odeint(fvdp,y0,x,tfirst=True,args=(1,1,5/3))
    
    y1, =plt.plot(x,y[:,0],label='y')
    y1_1, =plt.plot(x,y[:,1],label='y_one_differ')
    plt.legend(handles=[y1,y1_1])
    plt.show()
    return y
y=solve_second_order_ode()
# In[]

# In[]