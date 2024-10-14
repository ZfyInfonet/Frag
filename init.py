import networkx as nx
import os
import json
import pandas as pd
import numpy as np
import random
import util
import config
from data_structure import SFC, VNF, VNFLink


def get(topo_name, sfc_num, dataset_index, arrive_rate):
    g = getGraph(topo_name)
    return g, getSfcSet(sfc_num, g.number_of_nodes(), dataset_index), getArrivalOrder(sfc_num, arrive_rate)


def getTopoNodeAndEdgeNum(topo_name):
    with open(f'./topo/{topo_name}.json', 'r') as f:
        topo = json.loads(f.read())
    topo_nodes = topo['nodes']
    topo_edges = topo['edges']
    return len(topo_nodes), len(topo_edges)


def getCompleteGraph(topo_node_num):
    g = nx.complete_graph(topo_node_num)
    g = util.swapIdLabel(g)
    topo_nodes = g.nodes
    topo_edges = g.edges
    node_num = len(topo_nodes)
    G = nx.Graph(paths=np.array([[None for _ in range(node_num)] for _ in range(node_num)], dtype=list)
                 )
    for node_id in topo_nodes:
        cpu_capacity = config.node_cpu_capacity
        mem_capacity = config.node_mem_capacity
        G.add_node(node_id,
                   cpu_capacity=cpu_capacity * config.cpu_capacity_factor,
                   mem_capacity=mem_capacity * config.mem_capacity_factor,
                   cpu_residual=cpu_capacity * config.cpu_capacity_factor,
                   mem_residual=mem_capacity * config.mem_capacity_factor,
                   vnf_list=[],
                   cpu_history=[0],
                   mem_history=[0],
                   para_1=[0, 0],
                   para_2=[0, 0],
                   is_cpu_overload=False,
                   is_mem_overload=False,
                   neb_paths_edges=[[] for _ in range(config.neighborhood_field)],
                   neb_nodes=[[] for _ in range(config.neighborhood_field)]
                   )
    edge_id = 0
    for topo_edge in topo_edges:
        bw_capacity = config.edge_bw_capacity

        G.add_edge(topo_edge[0],
                   topo_edge[1],
                   edge_id=edge_id,
                   bw_capacity=bw_capacity * config.bw_capacity_factor,
                   bw_residual=bw_capacity * config.bw_capacity_factor,
                   delay=random.uniform(*config.edge_propagation_delay_range),
                   vnf_link_list=[],
                   bw_history=[0],
                   para_1=0,
                   para_2=0,
                   is_bw_overload=False
                   )
        edge_id += 1

    for src_node in G.nodes:
        for dst_node in range(src_node + 1, node_num + 1):
            all_paths = sorted(list(nx.all_simple_paths(G, src_node, dst_node,
                                                        cutoff=config.max_path_length)), key=len)
            G.graph['paths'][src_node - 1, dst_node - 1] = all_paths
            G.graph['paths'][dst_node - 1, src_node - 1] = all_paths

    G.graph['adj_matrices'] = getMultiHopAdjacentMatrices(G)
    return G


def getGraph(topo_name):
    # get topology
    with open(f'./topo/{topo_name}.json', 'r') as f:
        topo = json.loads(f.read())
    topo_nodes = topo['nodes']
    topo_edges = topo['edges']
    node_num = len(topo_nodes)
    G = nx.Graph(paths=np.array([[None for _ in range(node_num)] for _ in range(node_num)], dtype=list)
                 )
    for node_id in topo_nodes:
        cpu_capacity = config.node_cpu_capacity
        mem_capacity = config.node_mem_capacity
        G.add_node(node_id,
                   cpu_capacity=cpu_capacity * config.cpu_capacity_factor,
                   mem_capacity=mem_capacity * config.mem_capacity_factor,
                   cpu_residual=cpu_capacity * config.cpu_capacity_factor,
                   mem_residual=mem_capacity * config.mem_capacity_factor,
                   vnf_list=[],
                   cpu_history=[0],
                   mem_history=[0],
                   para_1=[0, 0],
                   para_2=[0, 0],
                   is_cpu_overload=False,
                   is_mem_overload=False,
                   neb_paths_edges=[[] for _ in range(config.neighborhood_field)],
                   neb_nodes=[[] for _ in range(config.neighborhood_field)]
                   )
    edge_id = 0
    for topo_edge in topo_edges:
        bw_capacity = config.edge_bw_capacity

        G.add_edge(topo_edge[0],
                   topo_edge[1],
                   edge_id=edge_id,
                   bw_capacity=bw_capacity * config.bw_capacity_factor,
                   bw_residual=bw_capacity * config.bw_capacity_factor,
                   delay=random.uniform(*config.edge_propagation_delay_range),
                   vnf_link_list=[],
                   bw_history=[0],
                   para_1=0,
                   para_2=0,
                   is_bw_overload=False
                   )
        edge_id += 1

    for src_node in G.nodes:
        for dst_node in range(src_node + 1, node_num + 1):
            all_paths = sorted(list(nx.all_simple_paths(G, src_node, dst_node,
                                                        cutoff=config.max_path_length)), key=len)
            G.graph['paths'][src_node - 1, dst_node - 1] = all_paths
            G.graph['paths'][dst_node - 1, src_node - 1] = all_paths

    G.graph['adj_matrices'] = getMultiHopAdjacentMatrices(G)
    return G


def getMultiHopAdjacentMatrices(g: nx.Graph):
    max_hop = config.neighborhood_field
    current_hop = 1
    already_neb = [[] for _ in range(g.number_of_nodes())]
    node_num = g.number_of_nodes()
    adj_matrices = []
    while current_hop <= max_hop:
        adj_matrix = np.zeros([node_num, node_num], dtype=int)
        for src in g.nodes:
            tmp_neb = []
            for dst in g.nodes:
                if src == dst:
                    adj_matrix[src - 1, dst - 1] = 0
                    continue
                all_paths = g.graph['paths'][src - 1][dst - 1]
                n_hop_paths = []
                for path in all_paths:
                    if len(path) > current_hop + 1:
                        break
                    if len(path) == current_hop + 1:
                        n_hop_paths.append(path)
                        if dst not in already_neb[src - 1]:
                            edges = util.getEdgesFromPath(path)
                            g.nodes[src]['neb_paths_edges'][current_hop - 1].append(edges)
                            g.nodes[src]['neb_nodes'][current_hop - 1].append(dst)
                            tmp_neb.append(dst)
                adj_matrix[src - 1, dst - 1] = len(n_hop_paths)
            already_neb[src - 1] += tmp_neb
        adj_matrices.append(adj_matrix)

        current_hop += 1
    return adj_matrices


def getSfcSet(sfc_num, node_num, dataset_index):
    if node_num < 2:
        raise Exception("Insufficient Node Number.")

    # choose dataset
    dataset_name = f'2013-{dataset_index + 7}'
    sfc_list = []
    for sfc_id in range(sfc_num):
        vnf_list = []
        virtual_vnf_list = []
        vnf_link_list = []
        # get trace
        trace_number = len([f for f in os.listdir(f'./dataset/{dataset_name}/')
                            if os.path.isfile(os.path.join(f'./dataset/{dataset_name}/', f))])
        trace_id = random.randint(1, trace_number)
        # avoid unmatched VNF, all VNFs of SFC use the same trace
        trace = pd.read_csv(f'./dataset/{dataset_name}/{trace_id}.csv',
                            usecols=['CPU usage [MHZ]', 'Memory usage [KB]',
                                     'Network transmitted throughput [KB/s]'],
                            delimiter=';\\t', engine='python', nrows=config.Time).to_numpy().T
        # choose the topology of VNF-FG from ./vnf_fg/
        vnf_fg_id = random.randint(1, 6)
        with open(f'./vnf_fg/vnf_fg_{vnf_fg_id}.json', 'r') as f:
            vnf_fg = json.loads(f.read())

        # create vnf instance
        for vnf_id in vnf_fg['vnfs']:

            if vnf_id == vnf_fg['entrance']:
                # create entrance vnf v_0
                virtual_vnf_list.append(VNF(
                    sfc_id,
                    vnf_id,
                    True,
                    False,
                    node_id=random.randint(1, node_num)
                ))
            # create exit vnfs, need nothing
            elif vnf_id in vnf_fg['exits']:
                virtual_vnf_list.append(VNF(
                    sfc_id,
                    vnf_id,
                    False,
                    True,
                    node_id=random.randint(1, node_num)
                ))
            # create true vnfs, need traces of cpu, memory, and bandwidth
            else:
                vnf_list.append(VNF(
                    sfc_id,
                    vnf_id,
                    False,
                    False,
                    trace_cpu=trace[0] * config.CPU_FACTOR,
                    trace_mem=trace[1] * config.MEM_FACTOR,
                    delay_proc=random.uniform(*config.vnf_proc_delay_range)
                ))
        # create vnf links
        for vnf_link in vnf_fg['links']:
            src_vnf_id = vnf_link[0]
            dst_vnf_id = vnf_link[1]
            src_vnf = None
            dst_vnf = None
            for vnf in vnf_list:
                if vnf.vnf_id == src_vnf_id:
                    src_vnf = vnf
                if vnf.vnf_id == dst_vnf_id:
                    dst_vnf = vnf
            for vnf in virtual_vnf_list:
                if vnf.vnf_id == src_vnf_id:
                    src_vnf = vnf
                if vnf.vnf_id == dst_vnf_id:
                    dst_vnf = vnf
            link = VNFLink(
                sfc_id,
                src_vnf_id,
                dst_vnf_id,
                (src_vnf, dst_vnf),
                random.uniform(*config.sfc_delay_limit_range),
                trace[2] * config.BW_FACTOR
            )
            src_vnf.downstream_link.append(link)
            src_vnf.links.append(link)
            dst_vnf.upstream_link = link
            dst_vnf.links.append(link)
            vnf_link_list.append(link)

        # create sfc
        sfc = SFC(sfc_id, vnf_list, virtual_vnf_list, vnf_link_list, random.uniform(*config.lifetime_range))
        sfc_list.append(sfc)
    return sfc_list


def getArrivalOrder(sfc_num, arrive_rate):
    sfc_count = 0
    sfc_arrival_order = []
    while sfc_count < sfc_num:
        arrived_sfc_num = np.random.poisson(arrive_rate)
        # if out of number
        if arrived_sfc_num + sfc_count > sfc_num:
            sfc_arrival_order.append(sfc_num - sfc_count)
            sfc_count += sfc_num - sfc_count
        else:
            sfc_arrival_order.append(arrived_sfc_num)
            sfc_count += arrived_sfc_num
    # avoid no sfc arrives at initial time
    while sfc_arrival_order[0] == 0:
        sfc_arrival_order.remove(0)
    return sfc_arrival_order
