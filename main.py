import init
import deploy
import torch
import numpy as np
from mig_alg import LBVMC, MOGA, Greedy, FaD, GAT, NoMig, Optimal
from util import *
from datetime import datetime
init_time = datetime.now()

gat_model_name = 'latest'

repeat_num = config.repeat_num
topology_name = config.topology_name
dataset_index = config.dataset_index
sfc_num = config.sfc_num
arrive_rate = config.arrive_rate
Time = config.Time
algorithms = [Baseline, Greedy, GAT, Optimal]
alg_name = ['Baseline', 'Greedy', 'GAT', 'Optimal']
gat_model = torch.load(f'./gat_models/{topology_name}/{gat_model_name}')
# evaluation metrics
accepted_sfc_ratio = np.zeros([repeat_num, len(algorithms)])
avr_deployed_sfc = np.zeros([repeat_num, len(algorithms)])
max_deployed_sfc = np.zeros([repeat_num, len(algorithms)])
migration_count = np.zeros([repeat_num, len(algorithms)])
migration_cost = np.zeros([repeat_num, len(algorithms)])
overload_count = np.zeros([repeat_num, len(algorithms)])
solved_overload_count = np.zeros([repeat_num, len(algorithms)])
cpu_var = np.zeros([repeat_num, len(algorithms)])
mem_var = np.zeros([repeat_num, len(algorithms)])
bw_var = np.zeros([repeat_num, len(algorithms)])
tot_var = np.zeros([repeat_num, len(algorithms)])
running_time = np.zeros([repeat_num, len(algorithms)])
frag_avr = np.zeros([repeat_num, len(algorithms)])
utility_function = np.zeros([repeat_num, len(algorithms)])
bw_use = np.zeros([repeat_num, len(algorithms)])
overload_situations = np.zeros([repeat_num, len(algorithms), Time])

for repeat_round in range(repeat_num):
    # initialization
    print(f' repeat round: {repeat_round}')
    g, sfc_list, sfc_arrive_order = init.get(topology_name, sfc_num, dataset_index, arrive_rate)
    # drawGraph(g)
    print('Initialization Over.')
    # print to check
    print("Nodes:", g.nodes())
    print("Edges:", g.edges())
    print("-------------")

    for i in range(0, len(algorithms)):
        alg = algorithms[i]
        g_cp = copy.deepcopy(g)
        sfc_list_cp = copy.deepcopy(sfc_list)
        sfc_id = 0
        arrived_sfc_count = 0
        overload_situation = np.zeros(Time)
        permission = np.zeros(Time)
        forbidden = False
        for t in range(Time):
            print(f' Time {t}:')
            # update resource utilization
            over_n, over_e, deployed_sfc_num, Var_old, bw_ratio, _ = updateGraphAndServices(g_cp, sfc_list_cp, t)
            if deployed_sfc_num > max_deployed_sfc[repeat_round, i]:
                max_deployed_sfc[repeat_round, i] = deployed_sfc_num
            if over_n or over_e:
                permission[t] = 1
                forbidden = True
                overload_count[repeat_round, i] += 1
                overload_situation[t] = overload_situation[t - 1] + 1
            else:
                overload_situation[t] = overload_situation[t - 1]
            bw_use[repeat_round, i] += bw_ratio
            cpu_var[repeat_round, i] += Var_old[0]
            mem_var[repeat_round, i] += Var_old[1]
            bw_var[repeat_round, i] += Var_old[2]
            tot_var[repeat_round, i] += Var_old[0] + Var_old[1] + Var_old[2]
            # print('Now:')
            # printNode(g_cp)
            # printEdge(g_cp)
            start_time = datetime.now()
            # run migration algorithm
            count, cost = alg.run(g_cp, t, over_n, over_e, GAT_model=gat_model)
            end_time = datetime.now()
            time_diff = end_time - start_time
            if time_diff.total_seconds() > running_time[repeat_round, i]:
                running_time[repeat_round, i] = time_diff.total_seconds()
            over_n, over_e, deployed_sfc_num, Var_old, bw_ratio, frag = updateGraphAndServices(g_cp, sfc_list_cp, t)
            frag_avr[repeat_round, i] += np.average(frag)
            # record running time
            # print('After Migration:')
            # printNode(g_cp)
            # printEdge(g_cp)
            migration_count[repeat_round, i] += count
            migration_cost[repeat_round, i] += cost

            if 1 in permission[t - min(t, config.forbidden_deploy_time): t]:
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
                        result = deploy.run(sfc, g_cp)
                        if not result:
                            print('Deployment fails.')
                        else:
                            accepted_sfc_ratio[repeat_round, i] += 1

                sfc_id += sfc_arrive_order[t]

            avr_deployed_sfc[repeat_round, i] += deployed_sfc_num
            print(f' cost: {cost}, deployed sfc: {deployed_sfc_num}')
        bw_use[repeat_round, i] /= Time
        avr_deployed_sfc[repeat_round, i] /= Time
        accepted_sfc_ratio[repeat_round, i] /= arrived_sfc_count
        overload_situations[repeat_round, i] = overload_situation
    for i in range(0, len(algorithms)):
        print(f' This is algorithm: {alg_name[i]}----------')
        print(f' accepted ratio: {accepted_sfc_ratio[repeat_round, i]}, \n'
              f' deployed sfc count: {avr_deployed_sfc[repeat_round, i]},\n'
              f' migration count: {migration_count[repeat_round, i]}, \n'
              f' migration cost: {"{:.2f}".format(migration_cost[repeat_round, i] / (1024 ** 2))}, \n'
              f' running time: {running_time[repeat_round, i]}, \n'
              f' utility function: {utility_function[repeat_round, i]}, \n'
              f' overload: {overload_situations[repeat_round, i][-1]}, \n'
              f' frag: {frag_avr[repeat_round, i]}, \n'
              )

    # calculate normalized utility function
    avr_dep_sfc = min_max_normalize(avr_deployed_sfc[repeat_round])
    mig_cost = min_max_normalize(migration_cost[repeat_round])

rep_avr_sfc = np.average(avr_deployed_sfc, axis=0)
rep_avr_count = np.average(migration_count, axis=0)
rep_avr_cost = np.average(migration_cost, axis=0) / (1024 ** 2)
rep_avr_cpu = np.average(cpu_var, axis=0)
rep_avr_mem = np.average(mem_var, axis=0)
rep_avr_bw = np.average(bw_var, axis=0)
rep_avr_var = np.average(tot_var, axis=0)
rep_avr_accept = np.average(accepted_sfc_ratio, axis=0)
rep_avr_runtime = np.average(running_time, axis=0)

rep_avr_bw_use = np.average(bw_use, axis=0)
overloads = np.average(overload_situations, axis=0)
overload_count = np.average(overload_count, axis=0)
rep_avr_frag = np.average(frag_avr, axis=0)
rep_avr_max_deployed = np.average(max_deployed_sfc, axis=0)
over_time = datetime.now()

print(f' This is algorithm: {alg_name}----------')
print(f' accepted ratio: {rep_avr_accept} , \n'
      f' deployed sfc count: {rep_avr_sfc},\n'
      f' migration count: {rep_avr_count}, \n'
      f' migration cost: {rep_avr_cost}, \n'
      f' running time: {rep_avr_runtime}, \n'
      f' utility function: {config.a1 * overload_count + config.a2 * rep_avr_cost}, \n'
      f' overload: {overload_count}, \n'
      f' frag: {rep_avr_frag}, \n'
      f' max deployed sfc: {rep_avr_max_deployed}, \n'
      )

print(f' Simulation begin time: {init_time}\n Simulation end time: {over_time}')
