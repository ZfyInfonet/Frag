import networkx as nx
import numpy as np
import util
import copy
import config


def run(g: nx.Graph, t, overload_nodes, overload_edges, sfc=None, input_model=None):
    mig_cost = 0
    mig_count = 0
    if overload_nodes:
        for node in overload_nodes:
            # remove the VNFs in the overloading nodes
            loop_count = 0
            black_list = []
            isOverload = util.isNodeOverload(g, node)
            while isOverload and loop_count < config.loop_limit and mig_count < config.mig_limit:
                loop_count += 1
                cpu_ratio = 1 - g.nodes[node]['cpu_residual'] / g.nodes[node]['cpu_capacity']
                mem_ratio = 1 - g.nodes[node]['mem_residual'] / g.nodes[node]['mem_capacity']

                if cpu_ratio >= mem_ratio:
                    max_vnf = None
                    max_cpu = 0
                    for vnf in g.nodes[node]['vnf_list']:
                        if vnf in black_list:
                            continue
                        if vnf.req_cpu > max_cpu:
                            max_vnf = vnf
                            max_cpu = vnf.req_cpu
                    bottleneck_id = 0
                else:
                    max_vnf = None
                    max_mem = 0
                    for vnf in g.nodes[node]['vnf_list']:
                        if vnf in black_list:
                            continue
                        if vnf.req_mem > max_mem:
                            max_vnf = vnf
                            max_mem = vnf.req_mem
                    bottleneck_id = 1
                if max_vnf is None:
                    break

                # get the destination node
                max_residual_resource = 0
                dst = None
                for n in g.nodes:
                    if n == node:
                        continue
                    if util.checkNodeResource(max_vnf, n, g):
                        if not bottleneck_id:
                            if g.nodes[n]['cpu_residual'] > max_residual_resource:
                                max_residual_resource = g.nodes[n]['cpu_residual']
                                dst = n
                        else:
                            if g.nodes[n]['mem_residual'] > max_residual_resource:
                                max_residual_resource = g.nodes[n]['mem_residual']
                                dst = n

                best_vnf = max_vnf
                if not dst:
                    black_list.append(best_vnf)
                    break

                # migration
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
