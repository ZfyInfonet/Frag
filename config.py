import numpy as np

# ---global---
repeat_num = 1000
arrive_rate = 10  # n / time slot
Time = 100
sfc_num = Time * arrive_rate
overload_th = 0.5
neighborhood_field = 5
forbidden_deploy_time = 0
topology_name = 'NSFNET'  # 'NSFNET' or 'USbackbone'
dataset_index = 0  # 0, 1, or 2
mig_limit = 9999999

# ---objective function---
a1 = 1
a2 = 1 - a1

# ---Fast Defragmentation Migration algorithm---
receptive_field = 2
receptive_weight = np.logspace(0, receptive_field, receptive_field+1, base=0.5)
neb_paths_limit = 20
loop_limit = 5

# Graph Attention Network---
gat_layer_num = 3

# ---genetic algorithm---
population_size = 10
crossover_probability = 0.2
crossover_swap_probability = 0.15
mutation_probability = 0.15
tau_1 = 10
tau_2 = 5
tau_3 = 10
evolution = 5
tot_evolution = 5

# ---graph---
max_path_length = 6

# ---edge---
edge_bw_capacity = 51200  # 50 MBps
edge_propagation_delay_range = [0.001, 0.005]  # 1 ms - 5 ms
edge_protected_bw = 1024  # 1 MBps
bw_capacity_factor = 1

# ---node---
node_cpu_capacity = 32000  # 2 GHz * 16
node_mem_capacity = 67108864  # 64 GB
cpu_capacity_factor = 1
mem_capacity_factor = 1

# ---sfc---
sfc_delay_limit_range = [0.020, 0.050]  # 20 ms - 50ms
lifetime_range = [1, 100]   # 1 - 100 time slot

# ---vnf---
vnf_proc_delay_range = [0.001, 0.005]  # s 1 ms - 5 ms
trace_num = 500  # 500 traces each dataset
CPU_FACTOR = 1
MEM_FACTOR = 1
BW_FACTOR = 1

# ---deployment algorithm---
max_path_number = 3


