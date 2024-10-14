import numpy as np
import pandas as pd

import config
import init
import deploy
import matplotlib.pyplot as plt
from util import *
from datetime import datetime

data_num = 1000
arrive_rate = 10
repeat_num = config.repeat_num
topology_name = config.topology_name
dataset_index = config.dataset_index
config.forbidden_deploy_time = 0

Time = 100
data_count = 0
result = np.zeros([data_num, 7])  # overload, frag, var

while data_count < data_num:
    for arr_rate in range(1, arrive_rate + 1):
        if data_count == data_num:
            break
        print(data_count)
        tmp_result = np.zeros([Time, 7])
        sfc_num = arr_rate * Time
        g, sfc_list, sfc_arrive_order = init.get(topology_name, sfc_num, dataset_index, arrive_rate)
        sfc_id = 0
        arrived_sfc_count = 0
        overload_situation = np.zeros(Time)
        permission = np.zeros(Time)
        forbidden = False
        for t in range(Time):
            # update resource utilization
            over_n, over_e, _, _, _, _ = updateGraphAndServices(g, sfc_list, t)
            if over_n or over_e:
                permission[t] = 1
                tmp_result[t, 0] = 1
                forbidden = True
                overload_situation[t] = overload_situation[t - 1] + 1
            else:
                overload_situation[t] = overload_situation[t - 1]
            if 1 in permission[t - min(t, config.forbidden_deploy_time): t]:
                forbidden = True
            else:
                forbidden = False
            # sfc arriving
            if t < len(sfc_arrive_order):
                # for each sfc arrived at time t
                for j in range(sfc_arrive_order[t]):
                    sfc = sfc_list[sfc_id + j]
                    sfc.arrive_time = t
                    arrived_sfc_count += 1
                    if not forbidden:
                        anything = deploy.run(sfc, g)

                sfc_id += sfc_arrive_order[t]
            _, _, _, util_set, util_max, frag = updateGraphAndServices(g, sfc_list, t)
            [cpu_util, mem_util, bw_util] = util_set
            var_set = np.array([np.var(cpu_util), np.var(mem_util), np.var(bw_util)])
            max_set = np.array([np.max(cpu_util), np.max(mem_util), np.max(bw_util)])
            avr_Var, max_Var = np.mean(var_set), np.max(var_set)
            avr_Frag, max_Frag = np.mean(frag), np.max(frag)
            avr_MaxUtil, max_MaxUtil = np.mean(max_set), np.max(max_set)

            tmp_result[t, 1] = avr_MaxUtil
            tmp_result[t, 2] = avr_Var
            tmp_result[t, 3] = avr_Frag
            tmp_result[t, 4] = max_MaxUtil
            tmp_result[t, 5] = max_Var
            tmp_result[t, 6] = max_Frag
        result[data_count] = np.average(tmp_result, axis=0)
        data_count += 1

result_T = result.T


df = pd.DataFrame(result)
df.to_excel(f'./result/sim3_data{data_num}_origin_alpha{config.receptive_field}.xlsx', index=False, header=False)

[x, y1, y2, y3, y4, y5, y6] = result_T.tolist()

fig, ax = plt.subplots()
ax.scatter(x, y1, label='Avr MaxUtil', color='blue', marker='o')
ax.scatter(x, y2, label='Avr Var', color='green', marker='v')
ax.scatter(x, y3, label='Avr Frag', color='cyan', marker='^')
ax.scatter(x, y4, label='Max MaxUtil', color='magenta', marker='<')
ax.scatter(x, y5, label='Max Var', color='orange', marker='>')
ax.scatter(x, y6, label='Max Frag', color='red', marker='*')

ax.legend()
plt.show()
