# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 17:58:16 2022

@author: zklei
"""

# In[]
import networkx as nx
import matplotlib.pyplot as plt
# In[]创建网络
nf=nx.Graph()
# In[]添加节点
nf.add_nodes_from(['sfo','lax','atl','jfx'])
print(nf.number_of_nodes())#节点数
# In[]添加边
nf.add_edges_from([('sfo','lax'),('jfx','lax'),('lax','atl')])
print(nf.number_of_edges())

# In[]绘制网络图
plt.figure(dpi=600)
pos=nx.spring_layout(nf)
nx.draw(nf,pos,with_labels=True)
plt.show()
# In[]
# =============================================================================
# print(nx.info(nf))
# print(nx.density(nf))
# print(nx.diameter(nf))
# print(nx.clustering(nf))
# print(list(nf.neighbors('sfo')))
# print(nx.degree_centrality(nf))
# print(nx.closeness_centrality(nf))
# print(nx.betweenness_centrality(nf))
# =============================================================================
# In[]
import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()
List=[(1,3,10),(1,4,60),(2,3,5),(2,4,20),(3,4,1)]
G.add_nodes_from(range(1,5))#先建立点
G.add_weighted_edges_from(List)#再赋予权值
w1=nx.to_numpy_matrix(G)#导出权重邻接边
w2=nx.get_edge_attributes(G,'weight')#导出赋权边的字典数据
pos=nx.spring_layout(G)
nx.draw(G,pos,with_labels=True)
nx.draw_networkx_edge_labels(G, pos,edge_labels=w2)
print("邻接矩阵:",w1)
print("字典:",G.adj)
print("列表",list(G.adjacency()))
plt.show()

# In[]创建
G=nx.Graph()
G.add_edge('A','B', weight=4)#带权边
G.add_edge('B','D', weight=2)
G.add_edge('A','C', weight=3)
G.add_edge('C','D', weight=5)
G.add_edge('A','D', weight=6)
G.add_edge('C','F', weight=7)
G.add_edge('A','G', weight=1)
G.add_edge('H','B', weight=2)
pos=nx.spring_layout(G)
# In[]绘制网络图
plt.figure(dpi=600)
nx.draw(G,pos,with_labels=True)
labels = nx.get_edge_attributes(G,'weight')#取边标签
nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)#绘制边上的标签
plt.show()
# In[]邻接矩阵生成
mat=nx.to_numpy_matrix(G)
print(mat)

# In[]两点间最短路
path=nx.dijkstra_path(G, source='H', target='F')
print(path)
distance=nx.dijkstra_path_length(G, source='H', target='F')
print(distance)

# In[]一点到所有点的最短路
p=nx.shortest_path(G, source='H')
d=nx.shortest_path_length(G,source='H')
for node in G.nodes():
    print("H 到",node,"的最短路径为:",p[node],end='\t')
    print("H 到",node,"的最短距离为:",d[node])
# In[]任意两点间的最短路距离
p=nx.shortest_path_length(G)
p=dict(p)
print(p)
for node1 in G.nodes():
    for node2 in G.nodes():
        print(node1,"到",node2,"的最短距离为:",p[node1][node2])
# In[]最短路的一个绘制
path=nx.dijkstra_path(G, source='H', target='F')
path_edges = list(zip(path, path[1:]))
plt.figure(dpi=600)
nx.draw(G,pos,with_labels=True)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red')
plt.show()

# In[]最小生成树
T=nx.minimum_spanning_tree(G)
print(sorted(T.edges(data=True)))

# In[]绘制生成树
mst=nx.minimum_spanning_edges(G,data=False)
edgelist=list(mst)
print(sorted(edgelist))
plt.figure(dpi=600)
nx.draw(G,pos,with_labels=True)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
nx.draw_networkx_edges(G, pos, edgelist=edgelist, edge_color='green')
plt.show()
# In[]A*算法
p=nx.astar_path(G,source='H',target='F')
print(p)
d=nx.astar_path_length(G,source='H',target='F')
print(d)
# In[]
#求0到5最短路
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
a=np.zeros([6,6])
a[0,1:]=[15,20,27,37,54]
a[1,2:]=[15,20,27,37];a[2,3:]=[16,21,28];
a[3,4:]=[16,21];a[4,5]=17
G=nx.DiGraph(a)
p=nx.shortest_path(G,0,5,weight='weight')
d=nx.shortest_path_length(G,0,5,weight='weight')
print("path=",p);print("d=",d)
# In[]
#最短路绘制
pos=nx.spring_layout(G)
plt.figure(dpi=600)
path_edges=list(zip(p,p[1:]))
nx.draw(G,pos,with_labels=True)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red')
plt.show()
# In[]
#floyd算法
# =============================================================================
# for(k=0;k<n;k++)//中转站0~k
#     for(i=0;i<n;i++) //i为起点
#         for(j=0;j<n;j++) //j为终点
#             if(d[i][j]>d[i][k]+d[k][j])//松弛操作 
#                 d[i][j]=d[i][k]+d[k][j]; 
# =============================================================================

# In[]
import networkx as nx
G=nx.DiGraph()
G.add_edges_from([('s','v1',{'capacity':8,'weight':2}),
                  ('s','v3',{'capacity':7,'weight':8}),
                  ('v1','v3',{'capacity':5,'weight':5}),
                  ('v1','v2',{'capacity':9,'weight':2}),
                  ('v3','v4',{'capacity':9,'weight':3}),
                  ('v2','v3',{'capacity':2,'weight':1}),
                  ('v4','t',{'capacity':10,'weight':7}),
                  ('v2','t',{'capacity':5,'weight':6}),
                  ('v4','v2',{'capacity':6,'weight':4})
                  ])
pos=nx.spring_layout(G)
pos['t'][0]=1;pos['t'][1]=0
pos['s'][0]=-1;pos['s'][1]=0
pos['v1'][0]=-0.33;pos['v1'][1]=1
pos['v2'][0]=0.33;pos['v2'][1]=1
pos['v3'][0]=-0.33;pos['v3'][1]=-1
pos['v4'][0]=0.33;pos['v4'][1]=-1
edge_label1=nx.get_edge_attributes(G, 'capacity')
edge_label2=nx.get_edge_attributes(G, 'weight')
edge_label={}
for i in edge_label1.keys():
    edge_label[i]=f'({edge_label1[i]:},{edge_label2[i]:})'
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_edge_labels(G, pos)
# In[]最小费用最大流
mincostFlow=nx.max_flow_min_cost(G, 's', 't')
mincost=nx.cost_of_flow(G,mincostFlow)

# In[]
import matplotlib.pyplot as plt
edge_label={}
for i in edge_label1.keys():
    edge_label[i]=f'({edge_label1[i]:},{edge_label2[i]:})'
for i in mincostFlow.keys():
    for j in mincostFlow[i].keys():
        edge_label[(i,j)]+=',F'+str(mincostFlow[i][j])
plt.figure(dpi=600)
nx.draw_networkx_nodes(G, pos,label="node")
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos,label="(容量,价格,流量)")        
nx.draw_networkx_edge_labels(G,pos,edge_label,font_size=12)
print(mincostFlow)
print(mincost)
plt.legend()
plt.xticks([])
# In[]
import geatpy as ea
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False
# TSP坐标点
places=[[35.0, 35.0], [41.0, 49.0], [35.0, 17.0], [55.0, 45.0], [55.0, 20.0], [15.0, 30.0],
        [25.0, 30.0], [20.0, 50.0], [10.0, 43.0], [55.0, 60.0], [30.0, 60.0], [20.0, 65.0], 
        [50.0, 35.0], [30.0, 25.0], [15.0, 10.0], [30.0, 5.0], [10.0, 20.0], [5.0, 30.0],
        [20.0, 40.0], [15.0, 60.0]]
#print(len(places))

# 定义问题
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = len(places) - 1 # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1] * Dim  # 决策变量下界
        ub = [Dim] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 新增一个属性存储旅行地坐标
        self.places = np.array(places)
    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen.copy()  # 得到决策变量矩阵
        # 添加最后回到出发地
        X = np.hstack([x, x[:, [0]]]).astype(int)
        ObjV = []  # 存储所有种群个体对应的总路程
        for i in range(pop.sizes):
            journey = self.places[X[i], :]  # 按既定顺序到达的地点坐标
            distance = np.sum(np.sqrt(np.sum(np.diff(journey.T) ** 2, 0)))  # 计算总路程
            ObjV.append(distance)
        pop.ObjV = np.array([ObjV]).T


"""调用模板求解"""
if __name__ == '__main__':
    """================================实例化问题对象============================"""
    problem = MyProblem()  # 生成问题对象
    """==================================种群设置=============================="""
    Encoding = 'P'  # 编码方式，采用排列编码
    NIND = 50  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    plt.figure(dpi=600)
    myAlgorithm = ea.soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 500 # 最大进化代数
    myAlgorithm.mutOper.Pm = 0.5  # 变异概率
    myAlgorithm.logTras = 0  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """===========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    BestIndi.save()  # 把最优个体的信息保存到文件中
    """==================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最短路程为：%s' % BestIndi.ObjV[0][0])
        print('最佳路线为：')
        best_journey = np.hstack([0, BestIndi.Phen[0, :], 0])
        for i in range(len(best_journey)):
            print(int(best_journey[i]), end=' ')
        print()
        # 绘图
        plt.figure(dpi=600)
        plt.plot(problem.places[best_journey.astype(int), 0], problem.places[best_journey.astype(int), 1], c='black')
        plt.plot(problem.places[best_journey.astype(int), 0], problem.places[best_journey.astype(int), 1], 'o',
                 c='black')
        for i in range(len(best_journey)):
            plt.text(problem.places[int(best_journey[i]), 0], problem.places[int(best_journey[i]), 1],
                     chr(int(best_journey[i]) + 65), fontsize=20)
        plt.grid(True)
        plt.xlabel('x坐标')
        plt.ylabel('y坐标')
    else:
        print('没找到可行解。')

# In[]
import networkx as nx
import matplotlib.pyplot as plt
from networkx import DiGraph
G = DiGraph()
for v in [1, 2, 3, 4, 5]:
    G.add_edge("Source", v, cost=10)
G.add_edge(1, 2, cost=10)
G.add_edge(2, 3, cost=10)
G.add_edge(3, 4, cost=15)
G.add_edge(4, 5, cost=10)
# In[]
pos=nx.spring_layout(G)
plt.figure(dpi=600)
nx.draw(G,pos,with_labels=True)
labels = nx.get_edge_attributes(G,'cost')
nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
plt.show()
# In[]
import networkx as nx
import matplotlib.pyplot as plt
from networkx import DiGraph
G = DiGraph()
for v in [1, 2, 3, 4, 5]:
    G.add_edge("Source", v, cost=10)
    G.add_edge(v, "Sink", cost=10)
G.add_edge(1, 2, cost=10)
G.add_edge(2, 3, cost=10)
G.add_edge(3, 4, cost=15)
G.add_edge(4, 5, cost=10)
# In[]
from vrpy import VehicleRoutingProblem
prob = VehicleRoutingProblem(G)
prob.num_stops = 3
prob.solve(time_limit=100)
print(prob.best_routes)
print(prob.best_value )
print(prob.best_routes_cost)  # 每条路径各自对应的cost
# In[]
a=prob.best_routes
a[1][3]='Source'
a[2][4]='Source'
G = DiGraph()
for v in [1, 2, 3, 4, 5]:
    G.add_edge("Source", v, cost=10)
G.add_edge(1, 2, cost=10)
G.add_edge(2, 3, cost=10)
G.add_edge(3, 4, cost=15)
G.add_edge(4, 5, cost=10)
pos=nx.spring_layout(G)
plt.figure(dpi=600)
nx.draw(G,pos,with_labels=True)
labels = nx.get_edge_attributes(G,'cost')
nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
edgelist_1=list(zip(a[1],a[1][1:]))
edgelist_2=list(zip(a[2],a[2][1:]))
nx.draw_networkx_edges(G, pos, edgelist=edgelist_1, edge_color='green')
nx.draw_networkx_edges(G, pos, edgelist=edgelist_2, edge_color='red')
plt.show()
# In[]
import networkx as nx
import matplotlib.pyplot as plt
from networkx import DiGraph
G = DiGraph()
for v in [1, 2, 3, 4, 5]:
    G.add_edge("Source", v, cost=10)
    G.add_edge(v, "Sink", cost=10)
G.add_edge(1, 2, cost=10)
G.add_edge(2, 3, cost=10)
G.add_edge(3, 4, cost=15)
G.add_edge(4, 5, cost=10)
# 定义需求量
for v in G.nodes():
    if v not in ["Source", "Sink"]:
        G.nodes[v]["demand"] = 5
# 容量约束
prob = VehicleRoutingProblem(G) # 原文档中缺少了这个，会报错，因为修改了G，prob需要重新定义
prob.load_capacity = 10
prob.solve()
print(prob.best_routes)
print(prob.best_value)
print("路程花费",prob.best_routes_cost)#路程花费
print("路程载货量",prob.best_routes_load)#路程载货量

# In[]
# 总时间约束
for (u, v) in G.edges():
    G.edges[u,v]["time"] = 20
G.edges[4,5]["time"] = 25
prob = VehicleRoutingProblem(G)
prob.duration = 60#总时间限制
prob.solve()
print(prob.best_value)
print(prob.best_routes)
print("路程花费",prob.best_routes_cost)#路程花费
print("路程载货量",prob.best_routes_load)#路程载货量
print("时间上限",prob.best_routes_duration)


# In[]时间窗
import networkx as nx
from vrpy import VehicleRoutingProblem

# Create graph
G = nx.DiGraph()
for v in [1, 2, 3, 4, 5]:
    G.add_edge("Source", v, cost=10, time=20)
    G.add_edge(v, "Sink", cost=10, time=20)
    G.nodes[v]["demand"] = 5
    G.nodes[v]["upper"] = 100
    G.nodes[v]["lower"] = 5
    G.nodes[v]["service_time"] = 1
G.nodes[2]["upper"] = 20
G.nodes["Sink"]["upper"] = 110
G.nodes["Source"]["upper"] = 100
G.add_edge(1, 2, cost=10, time=20)
G.add_edge(2, 3, cost=10, time=20)
G.add_edge(3, 4, cost=15, time=20)
G.add_edge(4, 5, cost=10, time=25)

# Create vrp
prob = VehicleRoutingProblem(G, num_stops=6, load_capacity=10, duration=64, time_windows=True)
prob.solve()
print(prob.best_routes)
print(prob.best_value)
print(prob.arrival_time)#到达时间
print(prob.departure_time)#离开时间
# In[]
from networkx import DiGraph
from vrpy import VehicleRoutingProblem
G = DiGraph()
for v in [1, 2, 3, 4, 5]:
    G.add_edge("Source", v, cost=[10, 11]) # 10是第一辆车的行驶成本，11是第二辆车的行驶成本，下面的以此类推
    G.add_edge(v, "Sink", cost=[10, 11])
    G.nodes[v]["demand"] = 5
G.add_edge(1, 2, cost=[10, 11])
G.add_edge(2, 3, cost=[10, 11])
G.add_edge(3, 4, cost=[15, 16])
G.add_edge(4, 5, cost=[10, 11])
prob=VehicleRoutingProblem(G, mixed_fleet=True, fixed_cost=[0, 5], load_capacity=[5, 20])
prob.solve()
print(prob.best_value)
print(prob.best_routes)
print(prob.best_routes_cost)
print(prob.best_routes_type)
print(prob.best_routes_load)#路程载货量
print(prob.best_routes_duration)

# In[]
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
mat=np.array([[0, 40, 55, 70, 80, 200, 110, 150, 70],
           [40, 0 ,65, 45, 110, 50, 75, 100, 110],
           [55, 65, 0 ,80, 100, 110, 80, 75, 75],
           [70, 45, 80, 0, 110, 50, 90, 80, 160],
           [80, 110, 100, 110, 0, 100, 80, 75, 100],
           [200, 50, 110, 50, 100, 0, 70, 90, 75],
           [110, 75, 80, 90, 80, 70, 0 ,80, 110],
           [150, 100, 75, 80, 75, 90, 80, 0 ,110],
           [70, 110, 75, 160, 100, 75, 110, 110, 0]]).astype('float64')
# In[]
G = nx.DiGraph()
for v in [1, 2, 3, 4, 5,6,7,8]:
    G.add_edge("Source", v,cost=mat[0,v],time=mat[0,v]/50)
    G.add_edge(v, "Sink",cost=mat[0,v],time=mat[v,0]/50)

for i in [1,2,3,4,5,6,7,8]:
    for j in [1,2,3, 4, 5,6,7,8]:
        if i!=j:
            G.add_edge(i,j,cost=mat[0,v],time=mat[i,j]/50)
e=np.array([2,5,1.5,5,3.5,2.5,5,1.5])
l=np.array([3,8,2,7,5,5,7,4])
g=np.array([1,2.5,3.5,3,2.5,3,3.5,3.5])
se=np.array([2,2,1.5,3.5,2.5,2,3,0.9])
for v in [1, 2, 3, 4, 5,6,7,8]:
    G.nodes[v]["demand"] = g[v-1]
    G.nodes[v]["upper"] = l[v-1]
    G.nodes[v]["lower"] = e[v-1]
    G.nodes[v]["service_time"] = se[v-1]
# In[]
from vrpy import VehicleRoutingProblem
prob = VehicleRoutingProblem(G,load_capacity=8,time_windows=True,fixed_cost=200)
prob.solve(max_iter=100)
print(prob.best_routes)
print(prob.best_value)
print(prob.arrival_time)#到达时间
print(prob.departure_time)#离开时间
# In[]
c=prob.best_routes
for i in range(1,len(c)+1):
    c[i][0]=0
    c[i][-1]=0
edge_1=list(zip(c[1],c[1][1:]))
edge_2=list(zip(c[2],c[2][1:]))
edge_3=list(zip(c[3],c[3][1:]))
edge_4=list(zip(c[4],c[4][1:]))
G=nx.Graph()
#导入所有边，每条边分别用tuple表示
G.add_edges_from([(0,6),(6,4),(4,0),(0,3),(3,2),(2,0),(0,1),(1,7),(7,0),(0,8),(8,5),(5,0)]) 
plt.figure(dpi=600)
pos=nx.spring_layout(G)
nx.draw(G,pos,with_labels=True)
nx.draw_networkx_edges(G, pos, edgelist=edge_1, edge_color='green')
nx.draw_networkx_edges(G, pos, edgelist=edge_2, edge_color='red')
nx.draw_networkx_edges(G, pos, edgelist=edge_3,edge_color='blue')
nx.draw_networkx_edges(G, pos, edgelist=edge_4,edge_color='purple')


# In[]
from random import*
import numpy as np
from math import*
from matplotlib import pyplot as plt

#随机初始化城市坐标
number_of_citys = 100
citys = []
for i in range(number_of_citys):
    citys.append([randint(1,100),randint(1,100)])
citys = np.array(citys)

#由城市坐标计算距离矩阵
distance = np.zeros((number_of_citys,number_of_citys))
for i in range(number_of_citys):
    for j in range(number_of_citys):
        distance[i][j] = sqrt((citys[i][0]-citys[j][0])**2+(citys[i][1]-citys[j][1])**2)

#初始化参数
iteration1 = 2000                #外循环迭代次数
T0 = 100000                      #初始温度，取大些
Tf = 1                           #截止温度，可以不用
alpha = 0.95                     #温度更新因子
iteration2 = 10                  #内循环迭代次数
fbest = 0                        #最佳距离

#初始化初解
x = []
for i in range(100):
    x.append(i)
np.random.shuffle(x)
x = np.array(x)
for j in range(len(x) - 1):
    fbest = fbest + distance[x[j]][x[j + 1]]
fbest = fbest + distance[x[-1]][x[0]]
xbest = x.copy()
f_now = fbest
x_now = xbest.copy()

for i in range(iteration1):
    for k in range(iteration2):
        #生成新解
        x1 = [0 for q in range(number_of_citys)]
        n1,n2 = randint(0,number_of_citys-1),randint(0,number_of_citys-1)
        n = [n1,n2]
        n.sort()
        n1,n2 = n
        #n1为0单独写
        if n1 > 0:
            x1[0:n1] = x_now[0:n1]
            x1[n1:n2+1] = x_now[n2:n1-1:-1]
            x1[n2+1:number_of_citys] = x_now[n2+1:number_of_citys]
        else:
            x1[0:n1] = x_now[0:n1]
            x1[n1:n2+1] = x_now[n2::-1]
            x1[n2+1:number_of_citys] = x_now[n2+1:number_of_citys]
        s = 0;
        for j in range(len(x1) - 1):
            s = s + distance[x1[j]][x1[j + 1]]
        s = s + distance[x1[-1]][x1[0]]
        #判断是否更新解
        if s <= f_now:
            f_now = s
            x_now = x1.copy()
        if s > f_now:
            deltaf = s - f_now
            if random() < exp(-deltaf/T0):
                f_now = s
                x_now = x1.copy()
        if s < fbest:
            fbest = s
            xbest = x1.copy()

    T0 = alpha * T0                #更新温度

    # if T0 < Tf:                  #停止准则为最低温度时可以取消注释
    #     break

#打印最佳路线和最佳距离
print(xbest)
print(fbest)

#绘制结果
plt.title('SA_TSP')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(citys[...,0],citys[...,1],'ob',ms = 3)
plt.plot(citys[xbest,0],citys[xbest,1])
plt.plot([citys[xbest[-1],0],citys[xbest[0],0]],[citys[xbest[-1],1],citys[xbest[0],1]],ms = 2)
plt.show()



# In[]
## 环境设定
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
import random

params = {
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)

from copy import deepcopy
#-----------------------------------
## 问题定义
creator.create('FitnessMin', base.Fitness, weights=(-1.0,)) # 最小化问题
# 给个体一个routes属性用来记录其表示的路线
creator.create('Individual', list, fitness=creator.FitnessMin) 

#-----------------------------------
## 个体编码
# 用字典存储所有参数 -- 配送中心坐标、顾客坐标、顾客需求、到达时间窗口、服务时间、车型载重量
dataDict = {}
# 节点坐标，节点0是配送中心的坐标
dataDict['NodeCoor'] = [(15,12), (3,13), (3,17), (6,18), (8,17), (10,14),
                           (14,13), (15,11), (15,15), (17,11), (17,16),
                            (18,19), (19,9), (19,21), (21,22), (23,9),
                            (23,22), (24,11), (27,21), (26,6), (26,9),
                            (27,2), (27,4), (27,17), (28,7), (29,14),
                            (29,18), (30,1), (30,8), (30,15), (30,17)
                           ]
# 将配送中心的需求设置为0
dataDict['Demand'] = [0,50,50,60,30,90,10,20,10,30,20,30,10,10,10,
                      40,51,20,20,20,30,30,30,10,60,30,20,30,40,20,20]
dataDict['MaxLoad'] = 400
dataDict['ServiceTime'] = 1

def genInd(dataDict = dataDict):
    '''生成个体， 对我们的问题来说，困难之处在于车辆数目是不定的'''
    nCustomer = len(dataDict['NodeCoor']) - 1 # 顾客数量
    perm = np.random.permutation(nCustomer) + 1 # 生成顾客的随机排列,注意顾客编号为1--n
    pointer = 0 # 迭代指针
    lowPointer = 0 # 指针指向下界
    permSlice = []
    # 当指针不指向序列末尾时
    while pointer < nCustomer -1:
        vehicleLoad = 0
        # 当不超载时，继续装载
        while (vehicleLoad < dataDict['MaxLoad']) and (pointer < nCustomer -1):
            vehicleLoad += dataDict['Demand'][perm[pointer]]
            pointer += 1
        if lowPointer+1 < pointer:
            tempPointer = np.random.randint(lowPointer+1, pointer)
            permSlice.append(perm[lowPointer:tempPointer].tolist())
            lowPointer = tempPointer
            pointer = tempPointer
        else:
            permSlice.append(perm[lowPointer::].tolist())
            break
    # 将路线片段合并为染色体
    ind = [0]
    for eachRoute in permSlice:
        ind = ind + eachRoute + [0]
    return ind
#-----------------------------------
## 评价函数
# 染色体解码
def decodeInd(ind):
    '''从染色体解码回路线片段，每条路径都是以0为开头与结尾'''
    indCopy = np.array(deepcopy(ind)) # 复制ind，防止直接对染色体进行改动
    idxList = list(range(len(indCopy)))
    zeroIdx = np.asarray(idxList)[indCopy == 0]
    routes = []
    for i,j in zip(zeroIdx[0::], zeroIdx[1::]):
        routes.append(ind[i:j]+[0])
    return routes

def calDist(pos1, pos2):
    '''计算距离的辅助函数，根据给出的坐标pos1和pos2，返回两点之间的距离
    输入： pos1, pos2 -- (x,y)元组
    输出： 欧几里得距离'''
    return np.sqrt((pos1[0] - pos2[0])*(pos1[0] - pos2[0]) + (pos1[1] - pos2[1])*(pos1[1] - pos2[1]))

#
def loadPenalty(routes):
    '''辅助函数，因为在交叉和突变中可能会产生不符合负载约束的个体，需要对不合要求的个体进行惩罚'''
    penalty = 0
    # 计算每条路径的负载，取max(0, routeLoad - maxLoad)计入惩罚项
    for eachRoute in routes:
        routeLoad = np.sum([dataDict['Demand'][i] for i in eachRoute])
        penalty += max(0, routeLoad - dataDict['MaxLoad'])
    return penalty

def calRouteLen(routes,dataDict=dataDict):
    '''辅助函数，返回给定路径的总长度'''
    totalDistance = 0 # 记录各条路线的总长度
    for eachRoute in routes:
        # 从每条路径中抽取相邻两个节点，计算节点距离并进行累加
        for i,j in zip(eachRoute[0::], eachRoute[1::]):
            totalDistance += calDist(dataDict['NodeCoor'][i], dataDict['NodeCoor'][j])    
    return totalDistance

def evaluate(ind):
    '''评价函数，返回解码后路径的总长度，'''
    routes = decodeInd(ind) # 将个体解码为路线
    totalDistance = calRouteLen(routes)
    return (totalDistance + loadPenalty(routes)),
#-----------------------------------
## 交叉操作
def genChild(ind1, ind2, nTrail=5):
    '''参考《基于电动汽车的带时间窗的路径优化问题研究》中给出的交叉操作，生成一个子代'''
    # 在ind1中随机选择一段子路径subroute1，将其前置
    routes1 = decodeInd(ind1) # 将ind1解码成路径
    numSubroute1 = len(routes1) # 子路径数量
    subroute1 = routes1[np.random.randint(0, numSubroute1)]
    # 将subroute1中没有出现的顾客按照其在ind2中的顺序排列成一个序列
    unvisited = set(ind1) - set(subroute1) # 在subroute1中没有出现访问的顾客
    unvisitedPerm = [digit for digit in ind2 if digit in unvisited] # 按照在ind2中的顺序排列
    # 多次重复随机打断，选取适应度最好的个体
    bestRoute = None # 容器
    bestFit = np.inf
    for _ in range(nTrail):
        # 将该序列随机打断为numSubroute1-1条子路径
        breakPos = [0]+random.sample(range(1,len(unvisitedPerm)),numSubroute1-2) # 产生numSubroute1-2个断点
        breakPos.sort()
        breakSubroute = []
        for i,j in zip(breakPos[0::], breakPos[1::]):
            breakSubroute.append([0]+unvisitedPerm[i:j]+[0])
        breakSubroute.append([0]+unvisitedPerm[j:]+[0])
        # 更新适应度最佳的打断方式
        # 将先前取出的subroute1添加入打断结果，得到完整的配送方案
        breakSubroute.append(subroute1)
        # 评价生成的子路径
        routesFit = calRouteLen(breakSubroute) + loadPenalty(breakSubroute)
        if routesFit < bestFit:
            bestRoute = breakSubroute
            bestFit = routesFit
    # 将得到的适应度最佳路径bestRoute合并为一个染色体
    child = []
    for eachRoute in bestRoute:
        child += eachRoute[:-1]
    return child+[0]

def crossover(ind1, ind2):
    '''交叉操作'''
    ind1[:], ind2[:] = genChild(ind1, ind2), genChild(ind2, ind1)
    return ind1, ind2

#-----------------------------------
## 突变操作
def opt(route,dataDict=dataDict, k=2):
    # 用2-opt算法优化路径
    # 输入：
    # route -- sequence，记录路径
    # 输出： 优化后的路径optimizedRoute及其路径长度
    nCities = len(route) # 城市数
    optimizedRoute = route # 最优路径
    minDistance = calRouteLen([route]) # 最优路径长度
    for i in range(1,nCities-2):
        for j in range(i+k, nCities):
            if j-i == 1:
                continue
            reversedRoute = route[:i]+route[i:j][::-1]+route[j:]# 翻转后的路径
            reversedRouteDist = calRouteLen([reversedRoute])
            # 如果翻转后路径更优，则更新最优解
            if  reversedRouteDist < minDistance:
                minDistance = reversedRouteDist
                optimizedRoute = reversedRoute
    return optimizedRoute

def mutate(ind):
    '''用2-opt算法对各条子路径进行局部优化'''
    routes = decodeInd(ind)
    optimizedAssembly = []
    for eachRoute in routes:
        optimizedRoute = opt(eachRoute)
        optimizedAssembly.append(optimizedRoute)
    # 将路径重新组装为染色体
    child = []
    for eachRoute in optimizedAssembly:
        child += eachRoute[:-1]
    ind[:] = child+[0]
    return ind,

#-----------------------------------
## 注册遗传算法操作
toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual, genInd)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', crossover)
toolbox.register('mutate', mutate)

## 生成初始族群
toolbox.popSize = 100
pop = toolbox.population(toolbox.popSize)

## 记录迭代数据
stats=tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('std', np.std)
hallOfFame = tools.HallOfFame(maxsize=1)

## 遗传算法参数
toolbox.ngen = 400
toolbox.cxpb = 0.8
toolbox.mutpb = 0.1

## 遗传算法主程序
## 遗传算法主程序
pop,logbook=algorithms.eaMuPlusLambda(pop, toolbox, mu=toolbox.popSize, 
                                      lambda_=toolbox.popSize,cxpb=toolbox.cxpb, mutpb=toolbox.mutpb,
                   ngen=toolbox.ngen ,stats=stats, halloffame=hallOfFame, verbose=True)


from pprint import pprint

def calLoad(routes):
    loads = []
    for eachRoute in routes:
        routeLoad = np.sum([dataDict['Demand'][i] for i in eachRoute])
        loads.append(routeLoad)
    return loads

bestInd = hallOfFame.items[0]
distributionPlan = decodeInd(bestInd)
bestFit = bestInd.fitness.values
print('最佳运输计划为：')
pprint(distributionPlan)
print('最短运输距离为：')
print(bestFit)
print('各辆车上负载为：')
print(calLoad(distributionPlan))

# 画出迭代图
minFit = logbook.select('min')
avgFit = logbook.select('avg')
plt.plot(minFit, 'b-', label='Minimum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')

# 计算结果
#最佳运输计划为：
#[[0, 9, 12, 19, 22, 24, 25, 17, 0],
# [0, 6, 4, 3, 2, 1, 5, 0],
# [0, 7, 0],
# [0, 8, 10, 11, 13, 14, 16, 18, 23, 26, 30, 29, 28, 27, 21, 20, 15, 0]]
#最短运输距离为：
#(136.93713103610511,)
#各辆车上负载为：
#[200, 290, 20, 391]




# In[]