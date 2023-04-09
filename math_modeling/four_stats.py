# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:14:20 2022

@author: zklei
"""
# In[]
#一、正态性检验
# In[]
#夏皮罗-威尔克测试 Shapiro-Wilk Test
#假设
#每个样本中的观测值都是独立和相同分布的（iid）。

from scipy.stats import shapiro
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat,p=shapiro(data)
print('stat=%.3f,p=%.3f'%(stat,p))
if p>0.05:
    print("Gaussian")
else:
    print("not Gaussian")
# In[]
#测试数据样本是否具有高斯分布。 D'Agostino and Pearson'
from scipy.stats import normaltest
#UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=10
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
stat, p = normaltest(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably Gaussian')
else:
    print('Probably not Gaussian')
# In[]
#安德森-达林检验 Anderson-Darling Test
#在每个显著性水平上检查是否满足正态分布
from scipy.stats import anderson
data = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
result = anderson(data)
print('stat=%.3f' % (result.statistic))
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print('Probably Gaussian at the %.1f%% level' % (sl))
    else:
        print('Probably not Gaussian at the %.1f%% level' % (sl))

# In[]
#二、相关性检验
# In[]
# 皮尔逊相关系数 Pearson’s Correlation Coefficient
# =============================================================================
# 检验两个样本是否有线性关系。 假设
# 每个样本中的观测值都是独立和相同分布的（iid）。
# 每个样本中的观测值都是正态分布。
# 每个样本中的观测值具有相同的方差。
# H0：两个样本是独立的。
# H1：样本之间有 dependency。
# =============================================================================
from scipy.stats import pearsonr
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat,p=pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')#不相关
else:
    print('Probably dependent')#不独立

# In[]
# 斯皮尔曼秩相关 Spearman’s Rank Correlation
# =============================================================================
# 检验两个样本是否有单调关系（monotonic relationship）。 假设
# 每个样本中的观测值都是独立的、同分布的（iid）。
# 每个样本中的观测值可以进行排序。
# =============================================================================
from scipy.stats import spearmanr
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')
# In[]
#Kendall’s Rank Correlation
from scipy.stats import kendalltau
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
stat, p = kendalltau(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')
else:
    print('Probably dependent')

# In[]
# 卡方检验 Chi-Squared Test
from scipy.stats import chi2_contingency
table = [[10, 20, 30],[6,  9,  17]]
stat, p, dof, expected = chi2_contingency(table)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably independent')#独立不相关
else:
    print('Probably dependent')
# In[]
#三、平稳性检验
#检验一个时间序列是否有单位根，例如是否有趋势或更普遍的自回归。 假设
#观察中是时间上的有序。
#H0：存在一个单位根（序列是非平稳的）。
#H1：不存在单位根（数列是静止的）。
from statsmodels.tsa.stattools import adfuller
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
stat, p, lags, obs, crit, t = adfuller(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably not Stationary')
else:
    print('Probably Stationary')
# In[]
# Kwiatkowski-Phillips-Schmidt-Shin
# =============================================================================
# 检验一个时间序列是否是趋势平稳的。 假设
# 观察中是时间上的有序。
# 解释
# 
# H0：时间序列不是趋势稳定的。
# H1：时间序列是趋势稳定的。
# =============================================================================
from statsmodels.tsa.stattools import kpss
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
stat, p, lags, crit = kpss(data)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably not Stationary')
else:
    print('Probably Stationary')
# In[]
# 四、参数统计假设检验 Parametric Statistical Hypothesis Tests
# In[]
# T检验 Student’s t-test
# H0：样本的均值相等。
# H1：样本的均值不相等。
import numpy as np
from scipy.stats import ttest_ind
data1 = np.array([0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869])
data2 = np.array([1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169])
stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')
# In[]
# =============================================================================
# 配对学生T检验 Paired Student’s t-test
# 检验两个配对样本的均值是否有显著差异。 假设
# 每个样本中的观测值都是独立和相同分布的（iid）。
# 每个样本中的观测值都是正态分布。
# 每个样本中的观测值具有相同的方差。
# 每个样本中的观测值都是成对的。
# =============================================================================
from scipy.stats import ttest_rel
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = ttest_rel(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')
# In[]
# 方差分析检验（ANOVA） Analysis of Variance Test (ANOVA)
# 检验两个或多个独立样本的均值是否有显著差异。
from scipy.stats import f_oneway
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = f_oneway(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

# In[]
# 重复计量方差分析检验 Repeated Measures ANOVA Test(无python)
# In[]
# 五、非参数统计假设检验 Nonparametric Statistical Hypothesis Tests(针对无法满足高斯分布的统计检验)
# In[]
# Mann-Whitney U Test
from scipy.stats import mannwhitneyu
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = mannwhitneyu(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

# In[]
# Wilcoxon Signed-Rank Test
# 每个样本中的观测值都是成对的。
from scipy.stats import wilcoxon
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = wilcoxon(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

# In[]
# Kruskal-Wallis H Test
# 此检验可用于确定两个以上的独立样本是否具有不同的分布。
from scipy.stats import kruskal
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [1, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
stat, p = kruskal(data1, data2,data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

# In[]
# Friedman Test
# 每个样本中的观测值都是成对的。(例如重复测量配对的样本,方差检验的重复测度分析或重复测度方差分析的非参数版本)
from scipy.stats import friedmanchisquare
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]
stat, p = friedmanchisquare(data1, data2, data3)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')


# In[]
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
import numpy as np
from scipy.stats import ks_2samp
a=np.random.poisson(10,1000)
b=np.random.poisson(10,1000)
print(ks_2samp(a,b))

# In[]


# In[]




