import networkx as nx
import numpy as np
import util
import config


def run(g: nx.Graph, t, overload_nodes, overload_edges, sfc=None, GAT_model = None):
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
                print(f' loop count: {loop_count}, node {node} needs migration')
                # print(g.nodes[node])
                best_vnf = util.getSuitableVnfToMig(g, node, black_list)
                if best_vnf is None:
                    break
                avr_req_bw = 0
                for link in best_vnf.downstream_link:
                    avr_req_bw += link.req_bw
                avr_req_bw += best_vnf.upstream_link.req_bw
                avr_req_bw /= (len(best_vnf.downstream_link) + 1)
                # get the destination node
                vnf_demand = [best_vnf.req_cpu, best_vnf.req_mem, avr_req_bw]
                dst = getMinFragNode(g, best_vnf, vnf_demand)
                if dst == node or not dst:
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
        for edge in overload_edges:
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
                    util.delVnfLinkMapping(link, g)
                    link.is_mapped = True
                    continue
                if src is None or dst is None:
                    continue
                paths = g.graph['paths'][src - 1, dst - 1]
                for path in paths:
                    path_flag = True
                    for e in util.getEdgesFromPath(path):
                        if e in overload_edges:
                            path_flag = False
                            break
                    if not path_flag:
                        continue
                    is_valid, _ = util.checkPathResource(link, path, g)
                    if is_valid:
                        util.migVnfLinkToPath(link, path, g)
                        removed_bw += link.req_bw
                        break

    # defragmentation

    return mig_count, mig_cost


def getFrag(g: nx.Graph, demand):
    r_f = config.receptive_field
    CON = np.zeros([g.number_of_nodes(), r_f + 1, 3])
    tot_cpu_used, tot_cpu_capacity, tot_mem_used, tot_mem_capacity, tot_bw_used, tot_bw_capacity = 0, 0, 0, 0, 0, 0
    for node in g.nodes:
        cpu_res, cpu_cap = g.nodes[node]['cpu_residual'], g.nodes[node]['cpu_capacity']
        mem_res, mem_cap = g.nodes[node]['mem_residual'], g.nodes[node]['mem_capacity']
        cpu_use, mem_use = cpu_cap - cpu_res, mem_cap - mem_res
        tot_cpu_used += cpu_use
        tot_cpu_capacity += cpu_cap
        tot_mem_used += mem_use
        tot_mem_capacity += mem_cap
        CON[node - 1, 0, 0], CON[node - 1, 0, 1] = cpu_res + 1, mem_res + 1
        CON[node - 1, 0, 2] = 1
        for hop in range(1, r_f + 1):
            path_list = g.nodes[node]['neb_paths_edges'][hop - 1]
            neb_nodes = g.nodes[node]['neb_nodes'][hop - 1]
            bw, count = float(0), int(0)
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
            cpu, mem = 0, 0
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
    Frag = np.array(demand) / CON
    ratios = np.array([tot_cpu_used / tot_cpu_capacity,
                       tot_mem_used / tot_mem_capacity,
                       tot_bw_used / tot_bw_capacity])

    weighted_frag = np.average(Frag, axis=2, weights=ratios)
    final_frag = np.average(weighted_frag, axis=1, weights=config.receptive_weight)

    return final_frag


def getMinFragNode(g: nx.Graph, vnf, vnf_demand):
    frag = getFrag(g, vnf_demand)
    min_frag_available_node = None
    min_frag = float('inf')
    for n in g.nodes:
        if util.checkNodeResource(vnf, n, g):
            if frag[n - 1] < min_frag:
                min_frag_available_node = n
                min_frag = frag[n - 1]

    return min_frag_available_node
