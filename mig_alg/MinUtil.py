import copy

import networkx as nx
import numpy as np
import util
import config
from data_structure import VNF, VNFLink, SFC


def run(g: nx.Graph, t, overload_nodes, overload_edges, sfc=None, input_model=None):
    mig_cost = 0
    mig_count = 0
    if overload_nodes:
        for node in overload_nodes:
            # remove the VNFs in the overloading nodes
            if mig_count >= config.mig_limit:
                break
            loop_count = 0
            black_list = []
            isOverload = util.isNodeOverload(g, node)
            while isOverload and loop_count < config.loop_limit and mig_count < config.mig_limit:
                loop_count += 1
                # print(g.nodes[node])
                best_vnf = util.getSuitableVnfToMig(g, node, black_list)
                if best_vnf is None:
                    break
                frag_list, cost_list = exhaustive_search_algorithm(g, best_vnf)
                util_function = config.a1 * frag_list + config.a2 * cost_list
                dst = None
                util_min = util_function[node - 1]
                for dst_n in g.nodes:
                    if dst_n == node:
                        continue
                    if util.checkNodeResource(best_vnf, dst_n, g):
                        if util_function[dst_n - 1] < util_min:
                            util_min = util_function[dst_n - 1]
                            dst = dst_n
                if not dst:
                    black_list.append(best_vnf)
                    break

                # migrate VNF and links
                mig_result, cost = util.migVnfAndLink(best_vnf, dst, g)
                if mig_result:
                    mig_cost += cost
                    mig_count += 1
                    isOverload = util.isNodeOverload(g, node)
                else:
                    black_list.append(best_vnf)

    if overload_edges:
        util.migOverloadEdges(g, overload_edges)

    return mig_count, mig_cost


def exhaustive_search_algorithm(g: nx.Graph, vnf: VNF):
    # create mini mirror graph
    n_info = {'node_list': [], 'cpu_list': [], 'mem_list': []}
    e_info = {'edge_list': [], 'bw_list': []}
    for n in g.nodes:
        n_info['node_list'].append(n)
        n_info['cpu_list'].append([g.nodes[n]['cpu_residual'], g.nodes[n]['cpu_capacity']])
        n_info['mem_list'].append([g.nodes[n]['mem_residual'], g.nodes[n]['mem_capacity']])
    for e in g.edges:
        e_info['edge_list'].append(e)
        e_info['bw_list'].append([g.edges[e]['bw_residual'], g.edges[e]['bw_capacity']])

    g_init = {'n_info': n_info, 'e_info': e_info}
    frag_init = util.getGraphFrag(g, config.receptive_field)
    frag_list = []
    cost_list = []
    for dst in g.nodes:
        if not util.checkNodeResource(vnf, dst, g):
            frag_list.append(np.max(frag_init))
            cost_list.append(0)
            continue
        path_list = []
        src = vnf.node_id
        for link in vnf.links:
            paths = g.graph['paths'][src - 1, dst - 1]
            for path in paths:
                is_valid, _ = util.checkPathResource(link, path, g)
                if is_valid:
                    path_list.append(path)
                    break
        if len(path_list) != len(vnf.links):
            frag_list.append(np.max(frag_init))
            cost_list.append(0)
            continue
        else:
            g_mirror = copy.deepcopy(g_init)
            for i in range(len(vnf.links)):
                link = vnf.links[i]
                path = path_list[i]
                new_edges = util.getEdgesFromPath(path)
                old_edges = link.edges
                for e in new_edges:
                    e_id = g.edges[e]['edge_id']
                    g_mirror['e_info']['bw_list'][e_id][0] -= link.req_bw
                for e in old_edges:
                    e_id = g.edges[e]['edge_id']
                    g_mirror['e_info']['bw_list'][e_id][0] += link.req_bw
            g_mirror['n_info']['cpu_list'][dst - 1][0] -= vnf.req_cpu
            g_mirror['n_info']['cpu_list'][src - 1][0] += vnf.req_cpu
            frag = util.getGraphMirrorFrag(g_mirror, config.receptive_field, g)
            cost = util.calMigrationCost(vnf, src, dst, g)
            frag_list.append(np.max(frag))
            cost_list.append(cost)
    return np.array(frag_list), np.array(cost_list)
