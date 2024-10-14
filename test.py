import pandas as pd
import numpy as np
import util
import copy
import torch
import networkx as nx
from torch import nn
import config
import json
import init

n_num = 20
g = nx.complete_graph(n_num)
g = util.swapIdLabel(g)
nodes = [node for node in g.nodes]
edges = [e for e in g.edges]

x = [1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8]
a, b, c, d = None, None, None, None
count = 0
total = 0
for a in range(8):
    for b in range(a + 1, 9):
        for c in range(b + 1, 10):
            for d in range(c + 1, 11):
                total += 1
                print(x[a], x[b], x[c], x[d])
                if x[a] == x[b] or x[a] == x[c] or x[a] == x[d] or x[b] == x[c] or x[b] == x[d] or x[c] == x[d]:

                    count += 1
print(count / total)
print(total)

import itertools

x = [1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8]
count = 0
total = 0

import itertools

x = [1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8]
count = 0
total = 0

# 使用 itertools.combinations 生成所有可能的 4 个值的组合
for comb in itertools.combinations(x, 4):
    total += 1
    print(comb)  # 打印组合的值
    if len(set(comb)) < 4:  # 如果集合长度小于4，表示有重复值
        count += 1

print(count / total)  # 出现至少一对相同数字的概率
print(total)  # 所有组合的总数
