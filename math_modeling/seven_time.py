# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 23:05:51 2022

@author: zklei
"""

# In[]
import pandas as pd
df=pd.read_csv("C:/Users/zklei/Desktop/股价数据.csv")
# In[]
import numpy as np
y=np.array([423,358,434,445,527,429,426,502,480,427,446])
def moveaverage(y,n):
    mt=['']*n
    for i in range(n+1,len(y)+2):
        m=y[i-n:i-1].mean()
        mt.append(round(m))
    return mt
yt3=moveaverage(y, 3)
yt5=moveaverage(y, 5)
s3=np.sqrt(((y[3:]-yt3[3:-1])**2).mean())
s5=np.sqrt(((y[5:]-yt5[5:-1])**2).mean())
print(yt3,s3)
print(yt5,s5)
d=pd.DataFrame(np.c_[np.r_[y,[' ']],np.r_[yt3],np.r_[yt5]])
print(d)
# In[]
def expmove(y,a):
    n=len(y)
    m=np.zeros(n)
    m[0]=y[0]
    for i in range(1,len(y)):
        m[i]=a*y[i]+(1-a)*m[i-1]
    return m
yt1=expmove(y, 0.2)
yt2=expmove(y, 0.5)
yt3=expmove(y, 0.8)
s1=np.sqrt(((y-yt1)**2).mean())
s2=np.sqrt(((y-yt2)**2).mean())
s3=np.sqrt(((y-yt3)**2).mean())
print(s1,s2,s3)
d=pd.DataFrame(np.c_[y,yt1,yt2,yt3])
print(d)
# In[]
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd

def gm11(x,n):
    nn=len(x)
    x1=np.cumsum(x)#第一次累加
    z1=(x1[:len(x1)-1]+x1[1:])*1.0/2.0 #紧邻均值
    yy=x[1:]
    xx=z1
    k = ((nn-1)*np.sum(xx*yy)-np.sum(xx)*np.sum(yy))/((nn-1)*np.sum(xx*xx)-np.sum(xx)*np.sum(xx))
    b = (np.sum(xx*xx)*np.sum(yy)-np.sum(xx)*np.sum(xx*yy))/((nn-1)*np.sum(xx*xx)-np.sum(xx)*np.sum(xx))
    a=-k
    print(a,b)
    
    if np.abs(a)>2:
        print("没有意义")
    elif np.abs(a)<=2:
        print("有意义")
        if 0.3<-a<=0.5:
            print("适用于短期预测")
        elif 0.5<-a<=0.8:
            print("对于短期数据谨慎使用!")
        elif 0.8<-a<=1.0:
            print("应该对此模型进行修正!")
        elif -a>1.0:
            print("不宜使用GM(1,1)模型预测!")
        else:
            print("适用于中期和长期预测")
            
    x0_hat=np.zeros(nn)
    x0_hat[0]=x[0]
    
    for m in range(nn-1):
        x0_hat[m+1]=(1-np.exp(a))*(x[0]-b/a)*np.exp(-a*m)
    result=np.zeros(n)
    
    for i in range(n):
        result[i]=(1-np.exp(a))*(x[0]-b/a)*np.exp(-a*(nn+i))
        
    absolute_residuals = x[1:] - x0_hat[1:]#计算绝对残差和相对残差
    relative_residuals = np.abs(absolute_residuals) / x[1:]
    
    class_ratio = x[1:]/x[:len(x)-1] #计算级比和级比偏差
    eta = np.abs(1-(1-0.5*a)/(1+0.5*a)*(1/class_ratio))
    
    return result,x0_hat,relative_residuals,eta

def metabolism_gm11(x0,predict_num):
    result=np.zeros(predict_num)
    for i in range(predict_num):
        temp=gm11(x0,1)
        result[i]=temp[0][0]
        x0=x0[1:]
        x0=np.append(x0,temp[0][0])
    return result,temp[1],temp[2],temp[3]


print("请输入一系列数据(少于10个):")
arr = input("")    
num = [float(n) for n in arr.split()]   
data=np.array(num)
print(data)


print("输入数据检验:")
data1=np.cumsum(data)
rho=data[1:]/data1[:-1]
print(f"光滑比例小于0.5的数据占比为{100*np.sum(rho<0.5)/(len(data)-1)}%")
print(f"除去前两个时期光滑比小于0.5的数据占比为{100*np.sum(rho[2:]<0.5)/(len(data)-3)}")
if 100*np.sum(rho<0.5)/(len(data)-1) >60  and 100*np.sum(rho[2:]<0.5)/(len(data)-3) > 90:
    print("可以")
else:
    print("不可以")


originlen=len(num)
second=np.arange(len(num)+3)+1
result1=gm11(data,3)
a=result1[0]

print(result1[0])
print(f"拟合效果评价{np.mean(result1[2])}")

if 0.1<np.mean(result1[2])<0.2:
    print("达到一般要求")
elif np.mean(result1[2])<=0.1:
    print("拟合效果非常不错!")
else:
    print("垃圾")
print(f"平均级比偏差{np.mean(result1[3])}")


if 0.1<np.mean(result1[3])<0.2:
    print("达到要求一般")
elif np.mean(result1[3])<=0.1:
    print("拟合效果非常不错!")
else:
    print("垃圾")
    
num1=num[:]
for i in range(len(a)):
    num.append(a[i])
print(num)

plt.plot(second,num)
plt.xlabel("样本数加预测周期数")
plt.ylabel("数值")
plt.title("灰色预测GM(1,1)原始版")
plt.grid()
plt.show()

result2=metabolism_gm11(data, 3)
b=result2[0]

print(result2[0])
print(f"拟合效果评价{np.mean(result2[2])}")

if 0.1<np.mean(result2[2])<0.2:
    print("达到一般要求")
elif np.mean(result2[2])<=0.1:
    print("拟合效果非常不错!")
else:
    print("垃圾")
    
    
print(f"平均级比偏差{np.mean(result2[3])}")
if 0.1<np.mean(result2[3])<0.2:
    print("达到要求一般")
elif np.mean(result2[3])<=0.1:
    print("拟合效果非常不错!")
else:
    print("垃圾")
    
for i in range(len(b)):
    num1.append(b[i])
print(num1)

plt.plot(second,num1)
plt.grid()
plt.xlabel("样本数加预测周期数")
plt.ylabel("数值")
plt.title("新陈代谢灰色预测GM(1,1)原始版")
plt.show()

# In[]灰色关联度分析
# 无量纲化
from matplotlib import pyplot as plt
import seaborn as sns
def dimensionlessProcessing(df_values,df_columns):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    res = scaler.fit_transform(df_values)
    return pd.DataFrame(res,columns=df_columns)

# 求第一列(影响因素)和其它所有列(影响因素)的灰色关联值
def GRA_ONE(data,m=0): # m为参考列
    # 标准化
    data = dimensionlessProcessing(data.values,data.columns)
    # 参考数列
    std = data.iloc[:,m]
    # 比较数列
    ce = data.copy()
    
    n = ce.shape[0]
    m = ce.shape[1]
    
    # 与参考数列比较，相减
    grap = np.zeros([n,m])
    for i in range(m):
        for j in range(n):
            grap[j,i] = abs(ce.iloc[j,i] - std[j])
            
    # 取出矩阵中的最大值和最小值
    mmax = np.amax(grap)
    mmin = np.amin(grap)
    ρ = 0.5 # 灰色分辨系数
    
    # 计算值
    grap = pd.DataFrame(grap).applymap(lambda x:(mmin+ρ*mmax)/(x+ρ*mmax))
    
    # 求均值，得到灰色关联值
    RT = grap.mean(axis=0)
    return pd.Series(RT)

# 调用GRA_ONE，求得所有因素之间的灰色关联值
def GRA(data):
    list_columns = np.arange(data.shape[1])
    df_local = pd.DataFrame(columns=list_columns)
    for i in np.arange(data.shape[1]):
        df_local.iloc[:,i] = GRA_ONE(data,m=i)
    return df_local
def ShowGRAHeatMap(data):
    # 色彩集
    colormap = plt.cm.RdBu
    plt.figure(figsize=(18,16),dpi=600)
    plt.title('Person Correlation of Features',y=1.05,size=18)
    sns.heatmap(data.astype(float),linewidths=0.1,vmax=1.0,square=True,\
               cmap=colormap,linecolor='white',annot=True)
    plt.show()

data=pd.read_csv("C:/Users/zklei/Desktop/2012/A/关系.csv")
data_gra = GRA(data)
ShowGRAHeatMap(data_gra)
# In[]
import pandas as pd
print(df.dtypes)
df.index = pd.to_datetime(df.date)
# In[]日收益率
import numpy as np
ts=df['close']
ts_ret = np.diff(ts,1)
ts_log = np.log(ts)
ts_diff = ts_log.diff(1)
# In[]
from statsmodels.tsa.stattools import adfuller
def adf_test(ts):
    adftest = adfuller(ts)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])

    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res
print(adf_test(ts))
print("---------------")
# In[]同上
from statsmodels.tsa.stattools import adfuller
a = df['close']
print(adfuller(a,    # 下述参数均为默认值
				maxlag=None, 
				regression='c', 
				autolag='AIC',   # 自动在[0, 1,...,maxlag]中间选择最优lag数的方法；
				store=False, 
				regresults=False)
				)
# In[]arma模型
from statsmodels.tsa.arima.model import ARMA
def draw_arma(ts,w):
    arma=ARIMA(ts,order=(w,0)).fit(disp=-1)
    ts_predict=arma.predict()
    
    plt.plot(ts_predict,label='predict')
    plt.plot(ts,label='origin')
    plt.legend(loc='best')
    plt.title('arma test w:%i' %w)
    print(arma.conf_int())
    return ts_predict
ts_predict=draw_arma(ts, 5)
# 结果如图所示：
# In[]
from pandas.plotting import lag_plot
lag_plot(a)   # 默认lag=1
plt.show()
# In[]相关性
from scipy.stats import pearsonr
b = a.shift(1)
print(pearsonr(a[1:], b[1:]))
# In[]acf,pacf
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(2,1,dpi=600)
plot_acf(a, ax=axes[0])
plot_pacf(a, ax=axes[1])
plt.tight_layout()
plt.show()
# In[]一二阶差分
plt.plot(ts,linewidth=2,color='navy')
plt.title('Original Series')

plt.plot(ts.diff(),linewidth=2)
plt.title('1st diff Series')


plt.plot(ts.diff().diff(),linewidth=2)
plot_acf(ts.diff().dropna())


plt.plot(ts.diff().diff(),linewidth=2)
plt.title('2st diff Series')


plot_acf(ts.diff().diff().dropna())

# In[]https://www.statsmodels.org/stable/statespace.html
from statsmodels.tsa.arima.model import ARIMA
# 1,1,2 ARIMA Model
model = ARIMA(a, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
print(model_fit.forecast(4))
# In[]
model_fit.plot_diagnostics(figsize=(20,10))
# In[]residual and density
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# In[]
predict_results = model_fit.get_prediction(start=973, end=977, dynamic=True)
predict_df = predict_results.summary_frame(alpha=0.10)
plt.plot(np.arange(880,973,1),a[880:973])
plt.plot(predict_df.index,predict_df['mean'])
plt.show()
# In[]
fig, ax = plt.subplots()
predict_df['mean'].plot(ax=ax)
ax.fill_between(predict_df.index, predict_df['mean_ci_lower'],
                predict_df['mean_ci_upper'], alpha=0.2)
# - Simulate two years of new data after the end of the sample
print(model_fit.simulate(8, anchor='end'))

# - Impulse responses for two years
print(model_fit.impulse_responses(8))

# In[]
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
model = pm.auto_arima(a, start_p=1, start_q=1,
                      information_criterion='aic',
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(model.summary())

# In[]




# In[]






# In[]