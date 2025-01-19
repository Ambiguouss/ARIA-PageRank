import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
n=100
p=0.1
alpha = 0.85
G = nx.gnp_random_graph(100,0.1,seed=0,directed=True)


A= nx.to_numpy_array(G)
S=A/A.sum(axis=1,keepdims=True)

M=(1-alpha)/n * np.ones((n,n))+alpha*S

u_list = []

u=np.full(n,1/n)
u_list.append(u)
for i in range(100):
    u=u@M
    u_list.append(u)


print(np.sum(u))
print(u)
print(u@M)