import networkx as nx
import matplotlib.pyplot as plt
import config
import numpy as np
import pandas as pd
import copy
from data_structure import SFC, VNF, VNFLink


def updateGraphAndServices(g: nx.Graph, sfc_list: [SFC], t: int):
    overload_nodes = []
    overload_edges = []
    deployed_SFCs = []
    delete_SFCs = []
    cpu_util, mem_util, bw_util = [], [], []
    bw_used = 0
    bw_capacity = 0
    # first update sfc
    for sfc in sfc_list:
        if sfc.arrive_time is None:
            continue
        undeployed_flag = False
        if t - sfc.arrive_time > sfc.lifetime:
            delete_SFCs.append(sfc)
            continue
        for vnf in sfc.vnf_list:
            vnf.req_cpu = vnf.trace_cpu[t]
            vnf.req_mem = vnf.trace_mem[t]
            if vnf.node_id is None:
                undeployed_flag = True
        for link in sfc.vnf_link_list:
            link.req_bw = link.trace_bw[t]
            if link.is_mapped is False:
                undeployed_flag = True
        if not undeployed_flag:
            sfc.is_mapped = True
            deployed_SFCs.append(sfc)
        else:
            sfc.is_mapped = False
            delete_SFCs.append(sfc)
    delSfcSetFromGraph(delete_SFCs, sfc_list, g)

    # then update graph
    for node in g.nodes:
        g.nodes[node]['cpu_residual'] = g.nodes[node]['cpu_capacity']
        g.nodes[node]['mem_residual'] = g.nodes[node]['mem_capacity']

        for vnf in g.nodes[node]['vnf_list']:
            g.nodes[node]['cpu_residual'] -= vnf.req_cpu
            g.nodes[node]['mem_residual'] -= vnf.req_mem
            if g.nodes[node]['cpu_residual'] < 0:
                g.nodes[node]['cpu_residual'] = 0
            if g.nodes[node]['mem_residual'] < 0:
                g.nodes[node]['mem_residual'] = 0
        g.nodes[node]['cpu_history'].append(g.nodes[node]['cpu_capacity'] - g.nodes[node]['cpu_residual'])
        g.nodes[node]['mem_history'].append(g.nodes[node]['mem_capacity'] - g.nodes[node]['mem_residual'])
        if 1 - g.nodes[node]['cpu_residual'] / g.nodes[node]['cpu_capacity'] > config.overload_th:
            g.nodes[node]['is_cpu_overload'] = True
        if 1 - g.nodes[node]['mem_residual'] / g.nodes[node]['mem_capacity'] > config.overload_th:
            g.nodes[node]['is_mem_overload'] = True
        if g.nodes[node]['is_cpu_overload'] or g.nodes[node]['is_mem_overload']:
            overload_nodes.append(node)
        cpu_util.append(1 - g.nodes[node]['cpu_residual'] / g.nodes[node]['cpu_capacity'])
        mem_util.append(1 - g.nodes[node]['mem_residual'] / g.nodes[node]['mem_capacity'])

    for edge in g.edges:
        g.edges[edge]['bw_residual'] = g.edges[edge]['bw_capacity']

        for link in g.edges[edge]['vnf_link_list']:
            g.edges[edge]['bw_residual'] -= link.req_bw
            if g.edges[edge]['bw_residual'] < 0:
                g.edges[edge]['bw_residual'] = 0
        g.edges[edge]['bw_history'].append(g.edges[edge]['bw_capacity'] - g.edges[edge]['bw_residual'])
        if 1 - g.edges[edge]['bw_residual'] / g.edges[edge]['bw_capacity'] > config.overload_th:
            g.edges[edge]['is_bw_overload'] = True
            overload_edges.append(edge)
        bw_util.append(1 - g.edges[edge]['bw_residual'] / g.edges[edge]['bw_capacity'])
        bw_used += g.edges[edge]['bw_history'][-1]
        bw_capacity += g.edges[edge]['bw_capacity']
    util_set = [cpu_util, mem_util, bw_util]
    util_max_set = [np.max(cpu_util), np.max(mem_util), np.max(bw_util)]
    util_max_max = max(util_max_set)
    return overload_nodes, overload_edges, len(deployed_SFCs), util_set, util_max_max, \
        getGraphFrag(g, config.receptive_field)


def calMigrationCost(vnf: VNF, src_node, dst_node, g: nx.Graph):
    if src_node == dst_node:
        return 0
    t_transmission = vnf.req_mem / config.edge_protected_bw
    if not g.graph['paths'][src_node - 1, dst_node - 1][0]:
        raise Exception('The graph is not connected.')
    edges = getEdgesFromPath(g.graph['paths'][src_node - 1, dst_node - 1][0])
    t_propagation = 0
    for edge in edges:
        t_propagation += g.edges[edge]['delay']
    t_total = t_propagation + t_transmission
    lost_traffic = vnf.upstream_link.req_bw
    lost_data = lost_traffic * t_total
    return lost_data


def migVnfAndLink(vnf: VNF, dst: int, g: nx.Graph):
    if vnf.node_id == dst:
        return True, 0
    if not checkNodeResource(vnf, dst, g):
        return False, 0

    path_list = []
    node = vnf.node_id
    for link in vnf.links:
        paths = g.graph['paths'][node - 1, dst - 1]
        for path in paths:
            is_valid, _ = checkPathResource(link, path, g)
            if is_valid:
                path_list.append(path)
                break
    if len(path_list) == len(vnf.links):
        for i in range(len(path_list)):
            migVnfLinkToPath(vnf.links[i], path_list[i], g)
        mig_result, cost = migVnfToNode(vnf, dst, g)
        return mig_result, cost
    else:
        return False, 0


def migOverloadEdges(g: nx.Graph, over_edges):
    for edge in over_edges:
        bw_res, bw_cap = g.edges[edge]['bw_residual'], g.edges[edge]['bw_capacity']
        bw_r = 1 - bw_res / bw_cap
        over_degree = (bw_r - config.overload_th) * bw_cap
        removed_bw = 0
        for link in g.edges[edge]['vnf_link_list']:
            if removed_bw > over_degree:
                break
            src_vnf, dst_vnf = link.vnfs[0], link.vnfs[1]
            src, dst = src_vnf.node_id, dst_vnf.node_id
            if src == dst:
                delVnfLinkMapping(link, g)
                link.is_mapped = True
                continue
            if src is None or dst is None:
                continue
            paths = g.graph['paths'][src - 1, dst - 1]
            for path in paths:
                path_flag = True
                for e in getEdgesFromPath(path):
                    if e in over_edges:
                        path_flag = False
                        break
                if not path_flag:
                    continue
                is_valid, _ = checkPathResource(link, path, g)
                if is_valid:
                    migVnfLinkToPath(link, path, g)
                    removed_bw += link.req_bw
                    break


def migVnfToNode(vnf: VNF, dst: int, g: nx.Graph):
    # begin migration
    if vnf.node_id == dst:
        return True, 0

    if vnf.node_id:
        if not checkNodeResource(vnf, dst, g):
            return False, 0
        else:
            origin_node = vnf.node_id
            delVnfFromNode(vnf, g)
            g.nodes[dst]['cpu_residual'] -= vnf.req_cpu
            g.nodes[dst]['mem_residual'] -= vnf.req_mem
            g.nodes[dst]['vnf_list'].append(vnf)
            vnf.node_id = dst

            cost = calMigrationCost(vnf, origin_node, dst, g)
            return True, cost
    else:
        return False, 0


def migVnfLinkToPath(vnf_link: VNFLink, path, g: nx.Graph):
    for e in vnf_link.edges:
        if vnf_link in g.edges[e]['vnf_link_list']:
            g.edges[e]['bw_residual'] += vnf_link.req_bw
            g.edges[e]['vnf_link_list'].remove(vnf_link)
    vnf_link.edges = []
    for e in getEdgesFromPath(path):
        vnf_link.edges.append(e)
        g.edges[e]['vnf_link_list'].append(vnf_link)
        g.edges[e]['bw_residual'] -= vnf_link.req_bw
    vnf_link.is_mapped = True


def checkNodeResource(vnf: VNF, node: int, g: nx.Graph):
    cpu_res, cpu_cap = g.nodes[node]['cpu_residual'], g.nodes[node]['cpu_capacity']
    mem_res, mem_cap = g.nodes[node]['mem_residual'], g.nodes[node]['mem_capacity']
    if cpu_res - vnf.req_cpu >= (1 - config.overload_th) * cpu_cap and \
            mem_res - vnf.req_mem >= (1 - config.overload_th) * mem_cap:
        return True
    else:
        return False


def checkSfcMapped(sfc: SFC, g: nx.Graph):
    sfc.is_mapped = True
    for vnf in sfc.vnf_list:
        if not vnf.node_id:
            sfc.is_mapped = False
    for link in sfc.vnf_link_list:
        if not link.is_mapped:
            sfc.is_mapped = False
    if not sfc.is_mapped:
        delSfcFromGraph(sfc, g)
    return sfc.is_mapped


def checkPathResource(vnf_link: VNFLink, path: [int], g: nx.Graph):
    edges = getEdgesFromPath(path)
    tot_prop_delay = 0
    bw_min = float('inf')
    for edge in edges:
        tot_prop_delay += g.edges[edge]['delay']
        bw_res = g.edges[edge]['bw_residual']
        if bw_res - vnf_link.req_bw < (1 - config.overload_th) * g.edges[edge]['bw_capacity']:
            return False, float('inf')
        if bw_res < bw_min:
            bw_min = bw_res

    if tot_prop_delay + vnf_link.vnfs[1].delay_proc > vnf_link.delay_limit:
        return False, float('inf')
    return True, bw_min


def getLinkAdjacentToVnf(vnf: VNF, sfc: SFC):
    links = []
    for link in sfc.vnf_link_list:
        if vnf.vnf_id in link.ends:
            links.append(link)
    return links


def addVnfToNode(vnf: VNF, node, g: nx.Graph):
    # print(vnf.vnf_id, vnf.node_id, vnf.is_entrance, vnf.is_exit, vnf.req_cpu, vnf.req_mem)
    if node is None:
        return False
    if checkNodeResource(vnf, node, g):
        g.nodes[node]['cpu_residual'] -= vnf.req_cpu
        g.nodes[node]['mem_residual'] -= vnf.req_mem
        g.nodes[node]['vnf_list'].append(vnf)
        vnf.node_id = node
        return True
    else:
        return False


def delVnfFromNode(vnf: VNF, g: nx.Graph):
    if vnf.node_id is None:
        return
    node = vnf.node_id
    if vnf in g.nodes[node]['vnf_list']:
        g.nodes[node]['cpu_residual'] += vnf.req_cpu
        g.nodes[node]['mem_residual'] += vnf.req_mem
        g.nodes[node]['vnf_list'].remove(vnf)
        vnf.node_id = None


def addVnfLinkToPath(vnf_link: VNFLink, path: [int], g: nx.Graph):
    edges = getEdgesFromPath(path)
    if checkPathResource(vnf_link, path, g):
        delVnfLinkMapping(vnf_link, g)
        for edge in edges:
            g.edges[edge]['bw_residual'] -= vnf_link.req_bw
            g.edges[edge]['vnf_link_list'].append(vnf_link)
            vnf_link.edges.append(edge)
        vnf_link.is_mapped = True
        return True
    else:
        return False


def delVnfLinkMapping(vnf_link: VNFLink, g: nx.Graph):
    for edge in vnf_link.edges:
        if vnf_link in g.edges[edge]['vnf_link_list']:
            g.edges[edge]['bw_residual'] += vnf_link.req_bw
            g.edges[edge]['vnf_link_list'].remove(vnf_link)
    vnf_link.edges = []
    vnf_link.is_mapped = False


def getEdgesFromPath(path):
    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return edges


def getVnfAvrReqBw(vnf: VNF):
    return sum([link.req_bw for link in vnf.links]) / len(vnf.links)


def calResourceUsageFromGraph(g: nx.graph):
    avr_cpu, avr_mem, avr_bw, var_cpu, var_mem, var_bw = 0, 0, 0, 0, 0, 0
    node_num = g.number_of_nodes()
    edge_num = g.number_of_edges()
    for node in g.nodes:
        avr_cpu += g.nodes[node]['cpu_history'][-1]
        avr_mem += g.nodes[node]['mem_history'][-1]
    avr_cpu /= node_num
    avr_mem /= node_num
    for edge in g.edges:
        avr_bw += g.edges[edge]['bw_history'][-1]
    avr_bw /= edge_num

    for node in g.nodes:
        var_cpu += ((g.nodes[node]['cpu_history'][-1] - avr_cpu) / g.nodes[node]['cpu_capacity']) ** 2
        var_mem += ((g.nodes[node]['mem_history'][-1] - avr_cpu) / g.nodes[node]['mem_capacity']) ** 2
    var_cpu /= node_num
    var_mem /= node_num

    for edge in g.edges:
        var_bw += ((g.edges[edge]['bw_history'][-1] - avr_bw) / g.edges[edge]['bw_capacity']) ** 2
    return var_cpu, var_mem, var_bw


def delSfcFromGraph(sfc: SFC, g: nx.Graph):
    for node in g.nodes:
        for vnf in g.nodes[node]['vnf_list']:
            if vnf.sfc_id == sfc.sfc_id:
                delVnfFromNode(vnf, g)
    for e in g.edges:
        for link in g.edges[e]['vnf_link_list']:
            if link.sfc_id == sfc.sfc_id:
                delVnfLinkMapping(link, g)
    sfc.is_mapped = False


def delSfcSetFromGraph(delete_sfc_set, sfc_list, g: nx.Graph):
    for node in g.nodes:
        for vnf in g.nodes[node]['vnf_list']:
            if sfc_list[vnf.sfc_id] in delete_sfc_set:
                delVnfFromNode(vnf, g)
    for e in g.edges:
        for link in g.edges[e]['vnf_link_list']:
            if sfc_list[link.sfc_id] in delete_sfc_set:
                delVnfLinkMapping(link, g)


def clearGraph(g: nx.Graph):
    for node in g.nodes:
        g.nodes[node]['vnf_list'] = []
        g.nodes[node]['cpu_residual'] = g.nodes[node]['cpu_capacity']
        g.nodes[node]['mem_residual'] = g.nodes[node]['mem_capacity']
    for edge in g.edges:
        g.edges[edge]['vnf_link_list'] = []
        g.edges[edge]['bw_residual'] = g.edges[edge]['bw_capacity']
    return g


def printGraph(g: nx.Graph):
    for node in g.nodes:
        print(f' node {node}:')
        print(g.nodes[node])
    for edge in g.edges:
        print(f' edge {edge}:')
        print(g.edges[edge])


def drawGraph(g: nx.Graph):
    nx.draw(g, with_labels=True)
    plt.show()


def isNodeOverload(g: nx.Graph, node):
    if 1 - g.nodes[node]['cpu_residual'] / g.nodes[node]['cpu_capacity'] <= config.overload_th:
        g.nodes[node]['is_cpu_overload'] = False
    if 1 - g.nodes[node]['mem_residual'] / g.nodes[node]['mem_capacity'] <= config.overload_th:
        g.nodes[node]['is_mem_overload'] = False
    if g.nodes[node]['is_cpu_overload'] or g.nodes[node]['is_mem_overload']:
        return True
    else:
        return False


def isNodeFutureOverload(g: nx.Graph, node, removed_req):
    cpu_pre = g.nodes[node]['para_1'][0] + g.nodes[node]['para_2'][0]
    mem_pre = g.nodes[node]['para_1'][1] + g.nodes[node]['para_2'][1]
    cpu_future_overload = True
    mem_future_overload = True
    if cpu_pre - removed_req[0] <= config.overload_th * g.nodes[node]['cpu_capacity']:
        cpu_future_overload = False
    if mem_pre - removed_req[1] <= config.overload_th * g.nodes[node]['mem_capacity']:
        mem_future_overload = False
    if cpu_future_overload or mem_future_overload:
        return True
    else:
        return False


def isEdgeOverload(g: nx.Graph, edge):
    if 1 - g.edges[edge]['bw_residual'] / g.edges[edge]['bw_capacity'] <= config.overload_th:
        g.edges[edge]['is_bw_overload'] = False
        return False
    else:
        g.edges[edge]['is_bw_overload'] = True
        return True


def getSuitableVnfToMig(g: nx.Graph, node, black_list):
    cpu_res, cpu_cap = g.nodes[node]['cpu_residual'], g.nodes[node]['cpu_capacity']
    mem_res, mem_cap = g.nodes[node]['mem_residual'], g.nodes[node]['mem_capacity']
    cpu_r, mem_r = 1 - cpu_res / cpu_cap, 1 - mem_res / mem_cap
    best_vnf = max_vnf = None
    min_degree = float('inf')
    max_degree = 0
    if cpu_r >= mem_r:
        over_degree = (cpu_r - config.overload_th) * cpu_cap
        for vnf in g.nodes[node]['vnf_list']:
            if vnf in black_list:
                continue
            if vnf.req_cpu > over_degree and vnf.req_cpu - over_degree < min_degree:
                min_degree = vnf.req_cpu - over_degree
                best_vnf = vnf
            if vnf.req_cpu > max_degree:
                max_degree = vnf.req_cpu
                max_vnf = vnf
        if best_vnf is None:
            best_vnf = max_vnf

    else:
        over_degree = (mem_r - config.overload_th) * mem_cap
        for vnf in g.nodes[node]['vnf_list']:
            if vnf in black_list:
                continue
            if vnf.req_mem > over_degree and vnf.req_mem - over_degree < min_degree:
                min_degree = vnf.req_mem - over_degree
                best_vnf = vnf
            if vnf.req_mem > max_degree:
                max_degree = vnf.req_mem
                max_vnf = vnf
        if best_vnf is None:
            best_vnf = max_vnf
    return best_vnf


def getGraphFrag(g: nx.Graph, receptive_field):
    CON = np.ones([g.number_of_nodes(), receptive_field + 1, 3])
    tot_cpu_used, tot_cpu_capacity, tot_mem_used, tot_mem_capacity, tot_bw_used, tot_bw_capacity = 0, 0, 0, 0, 0, 0
    for node in g.nodes:
        tot_cpu_used += g.nodes[node]['cpu_capacity'] - g.nodes[node]['cpu_residual']
        tot_cpu_capacity += g.nodes[node]['cpu_capacity']
        tot_mem_used += g.nodes[node]['mem_capacity'] - g.nodes[node]['mem_residual']
        tot_mem_capacity += g.nodes[node]['mem_capacity']
        if isNodeOverload(g, node):
            continue
        CON[node - 1, 0, 0], CON[node - 1, 0, 1] = g.nodes[node]['cpu_residual'] + 1, g.nodes[node]['mem_residual'] + 1
        for hop in range(1, receptive_field + 1):
            path_list = g.nodes[node]['neb_paths_edges'][hop - 1]
            neb_nodes = g.nodes[node]['neb_nodes'][hop - 1]
            bw, count = float(1), int(0)
            for path in path_list:
                bw_min = float('inf')
                for edge in path:
                    bw_res = g.edges[edge]['bw_residual']
                    if bw_res < bw_min:
                        bw_min = bw_res
                bw += bw_min
                count += 1
            if count:
                bw /= count
            CON[node - 1, hop, 2] = bw
            cpu, mem = 1, 1
            for n in neb_nodes:
                cpu += g.nodes[n]['cpu_residual']
                mem += g.nodes[n]['mem_residual']
            if len(neb_nodes):
                cpu /= len(neb_nodes)
                mem /= len(neb_nodes)
            CON[node - 1, hop, 0], CON[node - 1, hop, 1] = cpu, mem
    for e in g.edges:
        tot_bw_used += g.edges[e]['bw_capacity'] - g.edges[e]['bw_residual']
        tot_bw_capacity += g.edges[e]['bw_capacity']

    ratios = np.array([tot_cpu_used / tot_cpu_capacity, tot_mem_used / tot_mem_capacity, tot_bw_used / tot_bw_capacity])
    final_frag = getFragFromCon(CON, ratios)
    return final_frag


def getGraphMirrorFrag(g_info, receptive_field, g_topo: nx.Graph):
    nodes, cpus, mems = g_info['n_info']['node_list'], g_info['n_info']['cpu_list'], g_info['n_info']['mem_list']
    edges, bws = g_info['e_info']['edge_list'], g_info['e_info']['bw_list']
    CON = np.ones([len(g_info['n_info']['node_list']), receptive_field + 1, 3])
    tot_cpu_used, tot_cpu_capacity, tot_mem_used, tot_mem_capacity, tot_bw_used, tot_bw_capacity = 0, 0, 0, 0, 0, 0
    for node in nodes:
        tot_cpu_used += cpus[node - 1][1] - cpus[node - 1][0]
        tot_cpu_capacity += cpus[node - 1][1]
        tot_mem_used += mems[node - 1][1] - mems[node - 1][0]
        tot_mem_capacity += mems[node - 1][1]
        if (1 - (cpus[node - 1][0] / cpus[node - 1][1]) > config.overload_th or
                1 - (mems[node - 1][0] / mems[node - 1][1]) > config.overload_th):
            continue
        CON[node - 1, 0, 0], CON[node - 1, 0, 1] = cpus[node - 1][0] + 1, mems[node - 1][0] + 1
        for hop in range(1, receptive_field + 1):
            path_list = g_topo.nodes[node]['neb_paths_edges'][hop - 1]
            neb_nodes = g_topo.nodes[node]['neb_nodes'][hop - 1]
            bw, count = float(1), int(0)
            for path in path_list:
                bw_min = float('inf')
                for edge in path:
                    e_id = g_topo.edges[edge]['edge_id']
                    bw_res = bws[e_id][0]
                    if bw_res < bw_min:
                        bw_min = bw_res
                bw += bw_min
                count += 1
            if count:
                bw /= count
            CON[node - 1, hop, 2] = bw
            cpu, mem = 1, 1
            for n in neb_nodes:
                cpu += cpus[n - 1][0]
                mem += mems[n - 1][0]
            if len(neb_nodes):
                cpu /= len(neb_nodes)
                mem /= len(neb_nodes)
            CON[node - 1, hop, 0], CON[node - 1, hop, 1] = cpu, mem
    for e in g_topo.edges:
        edge_id = g_topo.edges[e]['edge_id']
        tot_bw_used += bws[edge_id][1] - bws[edge_id][0]
        tot_bw_capacity += bws[edge_id][1]
    ratios = np.array([tot_cpu_used / tot_cpu_capacity, tot_mem_used / tot_mem_capacity, tot_bw_used / tot_bw_capacity])
    final_frag = getFragFromCon(CON, ratios)
    return final_frag


def getFragFromCon(con, ratios):
    Frag = np.array([1, 1, 1]) / con
    if np.sum(ratios) == 0:
        weighted_frag = np.average(Frag, axis=2)
    else:
        weighted_frag = np.average(Frag, axis=2, weights=ratios)
    final_frag = np.average(weighted_frag, axis=1, weights=config.receptive_weight)
    return final_frag


def getLearningInput(g: nx.Graph, vnf: VNF):
    n_num = g.number_of_nodes()
    e_num = g.number_of_edges()
    vnf_avr_bw = getVnfAvrReqBw(vnf)
    x = np.zeros((n_num, 4))
    e_attr = np.zeros((e_num * 2, 2))
    for n in g.nodes:
        cpu_r, cpu_c = g.nodes[n]['cpu_residual'], g.nodes[n]['cpu_capacity']
        mem_r, mem_c = g.nodes[n]['mem_residual'], g.nodes[n]['mem_capacity']
        x[n - 1, 0] = vnf.req_cpu / cpu_c
        x[n - 1, 1] = vnf.req_mem / mem_c
        x[n - 1, 2] = cpu_r / cpu_c
        x[n - 1, 3] = mem_r / mem_c
    for e in g.edges:
        e_id = g.edges[e]['edge_id']
        e_r, e_c = g.edges[e]['bw_residual'], g.edges[e]['bw_capacity']
        e_attr[e_id * 2, 0] = vnf_avr_bw / e_c
        e_attr[e_id * 2 + 1, 0] = vnf_avr_bw / e_c
        e_attr[e_id * 2, 1] = e_r / e_c
        e_attr[e_id * 2 + 1, 1] = e_r / e_c
    return x, e_attr


def swapIdLabel(g):
    nodes = []
    edges = []
    nodes_id = dict()
    nodes_label = dict()
    for i, label in enumerate(g.nodes()):
        nodes_id[label] = i + 1
        nodes_label[i + 1] = label
        nodes.append(i + 1)
    for (u, v) in g.edges():
        edges.append((nodes_id[u], nodes_id[v]))
    new_graph = nx.Graph()
    new_graph.add_nodes_from(nodes)
    for node in nodes:
        new_graph.add_node(node, labels=node)
    new_graph.add_edges_from(edges)
    return new_graph


def plotData(x, data, fig_name):
    final_data = np.array([x] + data).T
    pd.DataFrame(final_data).to_excel(f'./result/{fig_name}.xlsx', index=False)


def normalize(data):
    value_max = np.max(data)
    value_min = np.min(data)
    if value_max == value_min:
        return data
    return (data - value_min) / (value_max - value_min)


def printNode(g: nx.Graph):
    for n in g.nodes:
        cpu, cpu_over = g.nodes[n]['cpu_residual'] / g.nodes[n]['cpu_capacity'], False
        mem, mem_over = g.nodes[n]['mem_residual'] / g.nodes[n]['mem_capacity'], False
        if cpu < 1 - config.overload_th:
            cpu_over = True
        if mem < 1 - config.overload_th:
            mem_over = True
        print(f'node {n}: \n\t'
              f'cpu overload: {cpu_over}, mem overload: {mem_over}, cpu: {round(cpu, 4)},\tmem: {round(mem, 4)}')


def printEdge(g: nx.Graph):
    for e in g.edges:
        e_id = g.edges[e]['edge_id']
        bw, bw_over = g.edges[e]['bw_residual'] / g.edges[e]['bw_capacity'], False
        if bw < 1 - config.overload_th:
            bw_over = True
        print(f'edge {e_id} {e}: \n\tbw overload: {bw_over},\tbw: {round(bw, 4)}')
