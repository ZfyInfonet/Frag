import pandas as pd

import config
import init
import deploy
import torch
import os
from mig_alg import Greedy, GAT, NoMig, LBVMC, MinMaxFrag
from util import *
from datetime import datetime
init_time = datetime.now()

print(os.getcwd())
gat_model_name = 'MHGAT_latest'


config.a1 = 0.9
config.a2 = 1 - config.a1
config.arrive_rate = 10
config.forbidden_deploy_time = 0
config.overload_th = 0.5
Time = config.Time
topology_name = 'NSFNET'
dataset_index = config.dataset_index
arrive_rate = config.arrive_rate
sfc_num = Time * config.arrive_rate
forbidden_deploy_time = config.forbidden_deploy_time
algorithms = [LBVMC, Greedy, GAT]
alg_name = ['LBVMC', 'Greedy', 'MHGAT']
gat_model = torch.load(f'./gat_models/{topology_name}/{gat_model_name}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gat_model.device = device

# x axis definition
repeat_num = 1000
x_list = [0.1, 0.316, 1, 3.16, 10]
result = np.zeros([len(x_list), 5, len(algorithms)])
result_name = 'bw_capacity_factor_9_14'
x_index = 0
runtime_list = np.zeros([5, len(algorithms)])
for x in x_list:
    config.bw_capacity_factor = x
    sfc_num = Time * config.arrive_rate
    # evaluation metrics
    accepted_sfc_ratio = np.zeros([repeat_num, len(algorithms)])
    avr_deployed_sfc = np.zeros([repeat_num, len(algorithms)])
    migration_count = np.zeros([repeat_num, len(algorithms)])
    migration_cost = np.zeros([repeat_num, len(algorithms)])
    overload_count = np.zeros([repeat_num, len(algorithms)])
    running_time = np.zeros([repeat_num, len(algorithms)])
    frag_avr = np.zeros([repeat_num, len(algorithms)])
    utility_function = np.zeros([repeat_num, len(algorithms)])
    overload_situations = np.zeros([repeat_num, len(algorithms), Time])
    for repeat_round in range(repeat_num):
        # initialization
        print(f' x: {x}, repeat round: {repeat_round}')
        g, sfc_list, sfc_arrive_order = init.get(topology_name, sfc_num, dataset_index, config.arrive_rate)
        ele_num = g.number_of_nodes() + g.number_of_edges()
        for i in range(0, len(algorithms)):
            alg = algorithms[i]
            g_cp = copy.deepcopy(g)
            sfc_list_cp = copy.deepcopy(sfc_list)
            sfc_id = 0
            arrived_sfc_count = 0
            overload_situation = np.zeros(Time)
            permission = np.zeros(Time)
            forbidden = False
            max_frag = 0
            for t in range(Time):
                # update resource utilization
                over_n, over_e, _, _, _, frag = updateGraphAndServices(g_cp, sfc_list_cp, t)
                if over_n or over_e:
                    permission[t] = 1
                    forbidden = True
                    overload_count[repeat_round, i] += (len(over_n) + len(over_e)) / (ele_num * Time)
                    overload_situation[t] = overload_situation[t - 1] + 1
                else:
                    overload_situation[t] = overload_situation[t - 1]

                start_time = datetime.now()
                # run migration algorithm

                count, cost = alg.run(g_cp, t, over_n, over_e, input_model=gat_model)
                end_time = datetime.now()
                time_diff = end_time - start_time
                if time_diff.total_seconds() > running_time[repeat_round, i]:
                    running_time[repeat_round, i] = time_diff.total_seconds()
                over_n, over_e, deployed_sfc_num, Var_old, bw_ratio, frag = updateGraphAndServices(g_cp, sfc_list_cp, t)
                frag_avr[repeat_round, i] += np.max(frag) / Time
                migration_count[repeat_round, i] += count
                migration_cost[repeat_round, i] += cost

                if 1 in permission[t - min(t, forbidden_deploy_time): t]:
                    forbidden = True
                else:
                    forbidden = False
                # sfc arriving
                if t < len(sfc_arrive_order):
                    # for each sfc arrived at time t
                    for j in range(sfc_arrive_order[t]):
                        sfc = sfc_list_cp[sfc_id + j]
                        sfc.arrive_time = t
                        arrived_sfc_count += 1
                        # print(f' sfc id: {sfc.sfc_id}')
                        if not forbidden:
                            deploy_result = deploy.run(sfc, g_cp)
                            if deploy_result:
                                accepted_sfc_ratio[repeat_round, i] += 1

                    sfc_id += sfc_arrive_order[t]

                avr_deployed_sfc[repeat_round, i] += deployed_sfc_num
                # print(f' cost: {cost}, deployed sfc: {deployed_sfc_num}')
            avr_deployed_sfc[repeat_round, i] /= Time
            accepted_sfc_ratio[repeat_round, i] /= arrived_sfc_count
            overload_situations[repeat_round, i] = overload_situation
        # for i in range(0, len(algorithms)):
        #     print(f' This is algorithm: {alg_name[i]}----------')
        #     print(f' accepted ratio: {accepted_sfc_ratio[repeat_round, i]}, \n'
        #           f' deployed sfc count: {avr_deployed_sfc[repeat_round, i]},\n'
        #           f' migration count: {migration_count[repeat_round, i]}, \n'
        #           f' migration cost: {"{:.2f}".format(migration_cost[repeat_round, i] / (1024 ** 2))}, \n'
        #           f' running time: {running_time[repeat_round, i]}, \n'
        #           f' utility function: {utility_function[repeat_round, i]}, \n'
        #           f' overload: {overload_situations[repeat_round, i][-1]}, \n'
        #           f' frag: {frag_avr[repeat_round, i]}, \n'
        #           )


    rep_avr_sfc = np.average(avr_deployed_sfc, axis=0)
    rep_avr_count = np.average(migration_count, axis=0)
    rep_avr_cost = np.average(migration_cost, axis=0) / (1024 ** 2)
    rep_avr_accept = np.average(accepted_sfc_ratio, axis=0)
    rep_avr_runtime = np.average(running_time, axis=0)
    overloads = np.average(overload_situations, axis=0)
    overload_count = np.average(overload_count, axis=0)
    rep_avr_frag = np.average(frag_avr, axis=0)
    over_time = datetime.now()
    print(f' This is algorithm: {alg_name}----------')
    print(f' accepted ratio: {rep_avr_accept} , \n'
          f' deployed sfc count: {rep_avr_sfc},\n'
          f' migration count: {rep_avr_count}, \n'
          f' migration cost: {rep_avr_cost}, \n'
          f' running time: {rep_avr_runtime}, \n'
          f' utility function: {config.a1 * rep_avr_frag + config.a2 * rep_avr_cost}, \n'
          f' overload: {overload_count}, \n'
          f' frag: {rep_avr_frag}, \n'
          )
    result[x_index] = np.array([overload_count, rep_avr_frag, rep_avr_cost, rep_avr_accept, config.a1*rep_avr_frag + (1-config.a1) * rep_avr_cost])
    runtime_list[x_index] = rep_avr_runtime
    x_index += 1

    print(f' Simulation begin time: {init_time}\n Simulation end time: {over_time}')


data_overload = pd.DataFrame(result[:, 0, :])
data_frag = pd.DataFrame(result[:, 1, :])
data_cost = pd.DataFrame(result[:, 2, :])
data_accept = pd.DataFrame(result[:, 3, :])
data_func = pd.DataFrame(result[:, 4, :])
print(f' over:{data_overload}')
print(f' frag:{data_frag}')
print(f' cost:{data_cost}')
print(f' acpt:{data_accept}')
print(f' func:{data_func}')


data_overload.to_excel(f' {result_name}_overload.xlsx', index=False, header=False)
data_frag.to_excel(f' {result_name}_frag.xlsx', index=False, header=False)
data_cost.to_excel(f' {result_name}_cost.xlsx', index=False, header=False)
data_accept.to_excel(f' {result_name}_accept.xlsx', index=False, header=False)
data_func.to_excel(f' {result_name}_func.xlsx', index=False, header=False)