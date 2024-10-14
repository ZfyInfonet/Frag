import networkx as nx
import util
import config
from data_structure import SFC, VNF, VNFLink


# Greedy Deployment
def run(sfc: SFC, g: nx.Graph):
    order = 0
    cannot_deployed_flag = False
    init_node = None
    for vnf in sfc.vnf_list:
        # the first VNF
        if order == 0:
            order += 1
            init_node = findSuitableNode(vnf, g)
            if init_node is None:
                cannot_deployed_flag = True
                break
            util.addVnfToNode(vnf, init_node, g)
            src = sfc.virtual_vnf_list[0].node_id
            if src == init_node:
                vnf.upstream_link.is_mapped = True
                continue
            path = findSuitablePath(vnf.upstream_link, src, init_node, g)
            if path is None:
                cannot_deployed_flag = True
                break
            util.addVnfLinkToPath(vnf.upstream_link, path, g)

        else:
            # other VNFs
            is_valid = util.checkNodeResource(vnf, init_node, g)
            if is_valid:
                util.addVnfToNode(vnf, init_node, g)
                src = vnf.upstream_link.vnfs[0].node_id
                if src == init_node:
                    vnf.upstream_link.is_mapped = True
                    continue
                path = findSuitablePath(vnf.upstream_link, src, init_node, g)
                if path is None:
                    cannot_deployed_flag = True
                    break
                util.addVnfLinkToPath(vnf.upstream_link, path, g)
            else:
                new_node = findSuitableNode(vnf, g, init_node)
                if new_node is None:
                    cannot_deployed_flag = True
                    break
                else:
                    util.addVnfToNode(vnf, new_node, g)
                    src = vnf.upstream_link.vnfs[0].node_id
                    if src == new_node:
                        vnf.upstream_link.is_mapped = True
                        continue
                    if src is None:
                        cannot_deployed_flag = True
                        break  # whyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
                    path = findSuitablePath(vnf.upstream_link, src, new_node, g)
                    if path is None:
                        cannot_deployed_flag = True
                        break
                    util.addVnfLinkToPath(vnf.upstream_link, path, g)

    if cannot_deployed_flag:
        util.delSfcFromGraph(sfc, g)
        return False
    for link in sfc.vnf_link_list:
        if not link.is_mapped:
            src, dst = link.vnfs[0].node_id, link.vnfs[1].node_id
            if not src or not dst:
                util.delSfcFromGraph(sfc, g)
                return False
            if src == dst:
                link.is_mapped = True
                continue
            path = findSuitablePath(link, src, dst, g)
            if path is None:
                util.delSfcFromGraph(sfc, g)
                return False
            util.addVnfLinkToPath(link, path, g)
    return util.checkSfcMapped(sfc, g)


def findSuitableNode(vnf: VNF, g: nx.Graph, init_node=None):
    neb = []
    if init_node:
        for nodes in g.nodes[init_node]['neb_nodes']:
            for n in nodes:
                neb.append(n)
        candidate_nodes = []
        max_cpu_node = None
        max_mem_node = None
        max_residual_cpu = 0
        max_residual_mem = 0
        # find all nodes IN THE NEIGHBORHOODS
        for node in neb:
            # get candidate nodes
            if util.checkNodeResource(vnf, node, g):
                if g.nodes[node]['cpu_residual'] > max_residual_cpu:
                    max_cpu_node = node
                    max_residual_cpu = g.nodes[node]['cpu_residual']
                if g.nodes[node]['mem_residual'] > max_residual_mem:
                    max_mem_node = node
                    max_residual_mem = g.nodes[node]['mem_residual']
                candidate_nodes.append(node)
            if max_mem_node is None and candidate_nodes:
                max_mem_node = candidate_nodes[0]
            if max_cpu_node is None and candidate_nodes:
                max_cpu_node = candidate_nodes[0]
        if len(candidate_nodes) == 0:
            dst_node = None
        elif len(candidate_nodes) == 1:
            dst_node = candidate_nodes[0]
        else:
            if max_cpu_node == max_mem_node:
                dst_node = max_cpu_node
            else:
                cpu_res_A, cpu_cap_A = g.nodes[max_cpu_node]['cpu_residual'], g.nodes[max_cpu_node]['cpu_capacity']
                mem_res_A, mem_cap_A = g.nodes[max_cpu_node]['mem_capacity'], g.nodes[max_cpu_node]['mem_capacity']
                cpu_res_B, cpu_cap_B = g.nodes[max_mem_node]['cpu_residual'], g.nodes[max_mem_node]['cpu_capacity']
                mem_res_B, mem_cap_B = g.nodes[max_mem_node]['mem_capacity'], g.nodes[max_mem_node]['mem_capacity']
                A_cpu_ratio = (cpu_cap_A - cpu_res_A + vnf.req_cpu) / cpu_cap_A
                A_mem_ratio = (mem_cap_A - mem_res_A + vnf.req_mem) / mem_cap_A
                B_cpu_ratio = (cpu_cap_B - cpu_res_B + vnf.req_cpu) / cpu_cap_B
                B_mem_ratio = (mem_cap_B - mem_res_B + vnf.req_mem) / mem_cap_B
                if abs(A_cpu_ratio - A_mem_ratio) > abs(B_cpu_ratio - B_mem_ratio):
                    dst_node = max_mem_node
                else:
                    dst_node = max_cpu_node
        if dst_node:
            return dst_node

    candidate_nodes = []
    max_cpu_node = None
    max_mem_node = None
    max_residual_cpu = 0
    max_residual_mem = 0
    # find all nodes that meet the resource constraint
    for node in g.nodes:
        # get candidate nodes
        if util.checkNodeResource(vnf, node, g):
            if g.nodes[node]['cpu_residual'] > max_residual_cpu:
                max_cpu_node = node
                max_residual_cpu = g.nodes[node]['cpu_residual']
            if g.nodes[node]['mem_residual'] > max_residual_mem:
                max_mem_node = node
                max_residual_mem = g.nodes[node]['mem_residual']
            candidate_nodes.append(node)
        if max_mem_node is None and candidate_nodes:
            max_mem_node = candidate_nodes[0]
        if max_cpu_node is None and candidate_nodes:
            max_cpu_node = candidate_nodes[0]
    if len(candidate_nodes) == 0:
        dst_node = None
    elif len(candidate_nodes) == 1:
        dst_node = candidate_nodes[0]
    else:
        if max_cpu_node == max_mem_node:
            dst_node = max_cpu_node
        else:
            cpu_res_A, cpu_cap_A = g.nodes[max_cpu_node]['cpu_residual'], g.nodes[max_cpu_node]['cpu_capacity']
            mem_res_A, mem_cap_A = g.nodes[max_cpu_node]['mem_capacity'], g.nodes[max_cpu_node]['mem_capacity']
            cpu_res_B, cpu_cap_B = g.nodes[max_mem_node]['cpu_residual'], g.nodes[max_mem_node]['cpu_capacity']
            mem_res_B, mem_cap_B = g.nodes[max_mem_node]['mem_capacity'], g.nodes[max_mem_node]['mem_capacity']
            A_cpu_ratio = (cpu_cap_A - cpu_res_A + vnf.req_cpu) / cpu_cap_A
            A_mem_ratio = (mem_cap_A - mem_res_A + vnf.req_mem) / mem_cap_A
            B_cpu_ratio = (cpu_cap_B - cpu_res_B + vnf.req_cpu) / cpu_cap_B
            B_mem_ratio = (mem_cap_B - mem_res_B + vnf.req_mem) / mem_cap_B
            if abs(A_cpu_ratio - A_mem_ratio) > abs(B_cpu_ratio - B_mem_ratio):
                dst_node = max_mem_node
            else:
                dst_node = max_cpu_node
    return dst_node


def findSuitablePath(vnf_link: VNFLink, src, dst, g: nx.Graph):
    paths = g.graph['paths'][src - 1, dst - 1]
    counter = 0
    selected_path_list = []
    used_bandwidth = []
    while counter < min([config.max_path_number, len(paths)]):
        path = paths[counter]
        counter += 1
        # check delay constraint
        is_valid, bw_min = util.checkPathResource(vnf_link, path, g)
        if not is_valid:
            continue
        # record the paths which meet the delay constraint
        selected_path_list.append(path)
        used_bandwidth.append(bw_min * (len(path) - 1))
    # select the path with the min used bandwidth
    if len(selected_path_list) == 0:
        return None
    min_index = used_bandwidth.index(min(used_bandwidth))
    path = selected_path_list[min_index]
    return path
