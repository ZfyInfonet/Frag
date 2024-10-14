import networkx as nx
import numpy as np
import util
import config

# prediction parameter
alpha = 0.5
beta = 0.5


def run(g: nx.Graph, t, overload_nodes, overload_edges, sfc=None, input_model = None):
    migration_total_cost = 0
    migration_count = 0
    load_predicted = predictLoad(g)
    over = np.zeros(len(g.nodes))
    under = np.zeros(len(g.nodes))
    F = []

    # Algorithm 1 PM Load State Classification Algorithm
    src_host = []
    dst_host = []
    for node in g.nodes:
        cpu_ratio = 1 - g.nodes[node]['cpu_residual'] / g.nodes[node]['cpu_capacity']
        mem_ratio = 1 - g.nodes[node]['mem_residual'] / g.nodes[node]['mem_capacity']
        if cpu_ratio >= mem_ratio:
            bottleneck_load = g.nodes[node]['cpu_capacity'] - g.nodes[node]['cpu_residual']
            predicted_load = load_predicted[node - 1, 0]
            Th = g.nodes[node]['cpu_capacity'] * config.overload_th

            if min([bottleneck_load, predicted_load]) > Th:
                over[node - 1] = calculateOver(bottleneck_load, predicted_load, Th)
            elif max([bottleneck_load, predicted_load]) < Th:
                under[node - 1] = calculateUnder(bottleneck_load, predicted_load, Th)

            if over[node - 1] == 1.0:
                src_host.append(node)
            elif 0 < over[node - 1] < 1:
                src_host.append(node)
            elif 0 < under[node - 1] < 1:
                dst_host.append(node)
            elif under[node - 1] == 1.0:
                dst_host.append(node)

        else:
            bottleneck_load = g.nodes[node]['mem_history'][-1]
            predicted_load = load_predicted[node - 1, 1]
            Th = g.nodes[node]['mem_capacity'] * config.overload_th

            if min([bottleneck_load, predicted_load]) > Th:
                over[node - 1] = calculateOver(bottleneck_load, predicted_load, Th)
            elif max([bottleneck_load, predicted_load]) < Th:
                under[node - 1] = calculateUnder(bottleneck_load, predicted_load, Th)

            if over[node - 1] == 1.0:
                src_host.append(node)
            elif 0 < over[node - 1] < 1:
                src_host.append(node)
            elif 0 < under[node - 1] < 1:
                dst_host.append(node)
            elif under[node - 1] == 1.0:
                dst_host.append(node)
        U_ratio = (cpu_ratio + mem_ratio) / 2
        balance = np.sqrt(
            ((cpu_ratio / (U_ratio + 0.01) - 1) ** 2 + (mem_ratio / (U_ratio + 0.01) - 1) ** 2) / 2 + 0.0001)
        F.append((max([over[node - 1], under[node - 1]]) * balance, node))

    # Algorithm 2 Resource Weight VM Selection Algorithm

    # sort sourceHost in order of decreasing F(i)
    sorted_src_host = [y[1] for y in sorted(F, key=lambda x: x[0], reverse=True)]
    selected_v = []
    maxQ = -9999
    for node in sorted_src_host:
        residual_cpu = g.nodes[node]['cpu_residual']
        residual_mem = g.nodes[node]['mem_residual']
        cpu_overload = residual_cpu < (1 - config.overload_th) * g.nodes[node]['cpu_capacity']
        mem_overload = residual_mem < (1 - config.overload_th) * g.nodes[node]['mem_capacity']
        if cpu_overload or mem_overload:
            is_cpu_overload = calculateTau(cpu_overload)
            is_mem_overload = calculateTau(mem_overload)
            w_c = is_cpu_overload * (1 - residual_cpu / g.nodes[node]['cpu_capacity'])
            w_m = is_mem_overload * (1 - residual_mem / g.nodes[node]['mem_capacity'])
            for vnf in g.nodes[node]['vnf_list']:
                vnf_cpu_ratio = vnf.req_cpu / g.nodes[node]['cpu_capacity']
                vnf_mem_ratio = vnf.req_mem / g.nodes[node]['mem_capacity']
                Q = w_c * ((vnf_cpu_ratio + 0.00001) ** is_cpu_overload) + \
                    w_m * ((vnf_mem_ratio + 0.00001) ** is_mem_overload)
                if Q > maxQ:
                    selected_v.append((vnf, Q))
                    maxQ = Q
        # since we do not consider the under-load, rest part is ignored

    # Algorithm 3 Resource Fitness and Load Correlation VM Placement Algorithm
    selected_v = [x[0] for x in sorted(selected_v, key=lambda y: y[1], reverse=True)]
    for vnf in selected_v:
        maxF = -9999
        dst = None
        dst_sort = []
        for node in dst_host:
            dst_sort.append((node, under[node - 1]))
        dst_host = [x[0] for x in sorted(dst_sort, key=lambda y: y[1], reverse=True)]
        for node in dst_host:
            if util.checkNodeResource(vnf, node, g):
                fit = calculateFit(vnf, node, g)
                cor_list = []
                for vm in g.nodes[node]['vnf_list']:
                    cor_list.append(calculateCorr(vnf, vm, t))
                f = fit - sum(cor_list)
                if f > maxF:
                    dst = node
                    maxF = f
        if dst is not None and vnf.node_id is not None:
            # migration
            path_list = []
            for link in vnf.links:
                paths = g.graph['paths'][vnf.node_id - 1, dst - 1]
                for path in paths:
                    is_valid, _ = util.checkPathResource(link, path, g)
                    if is_valid:
                        path_list.append(path)
                        break
            if len(path_list) == len(vnf.links):
                for i in range(len(path_list)):
                    util.migVnfLinkToPath(vnf.links[i], path_list[i], g)
                mig_result, cost = util.migVnfToNode(vnf, dst, g)
                migration_total_cost += cost
                migration_count += 1

    return migration_count, migration_total_cost


def predictLoad(g_cp: nx.Graph):
    #
    node_predicted = np.zeros([len(g_cp.nodes), 2])
    for node in g_cp.nodes:
        cpu_history = g_cp.nodes[node]['cpu_history']
        mem_history = g_cp.nodes[node]['mem_history']
        if len(cpu_history) == 1:
            node_predicted[node - 1, 0] = cpu_history[0]
            node_predicted[node - 1, 1] = mem_history[0]

            g_cp.nodes[node]['para_1'] = [cpu_history[0], mem_history[0]]
            g_cp.nodes[node]['para_2'] = [0, 0]
        else:
            s_tp1 = g_cp.nodes[node]['para_1']
            b_tp1 = g_cp.nodes[node]['para_2']
            s_t = [alpha * cpu_history[-1] + (1 - alpha) * (s_tp1[0] + b_tp1[0]),
                   alpha * mem_history[-1] + (1 - alpha) * (s_tp1[1] + b_tp1[1])]
            b_t = [beta * (s_t[0] - s_tp1[0]) + (1 - beta) * b_tp1[0],
                   beta * (s_t[1] - s_tp1[1]) + (1 - beta) * b_tp1[1]]
            node_predicted[node - 1, 0] = s_t[0] + b_t[0]
            node_predicted[node - 1, 1] = s_t[1] + b_t[1]

            g_cp.nodes[node]['para_1'] = s_t
            g_cp.nodes[node]['para_2'] = b_t
    return node_predicted


def calculateOver(L, PL, Th):
    if L >= Th and PL > Th:
        return 1.0
    elif L <= Th and PL < Th:
        return 0
    else:
        return max([L, PL]) / abs(L - PL)


def calculateUnder(L, PL, Th):
    if L <= Th and PL < Th:
        return 1.0
    elif L >= Th and PL > Th:
        return 0
    else:
        return max([L, PL]) / abs(L - PL)


def calculateTau(x):
    if x:
        return 1
    else:
        return -1


def calculateCorr(vnf1, vnf2, t):
    cpu_numerator = []
    cpu_denominator_1 = []
    cpu_denominator_2 = []
    mem_numerator = []
    mem_denominator_1 = []
    mem_denominator_2 = []
    if t < 5:
        for tmp_t in range(t + 1):
            x_c = vnf1.trace_cpu[tmp_t]
            x_m = vnf1.trace_mem[tmp_t]
            x_avr = (x_c + x_m) / 2
            y_c = vnf2.trace_cpu[tmp_t]
            y_m = vnf2.trace_mem[tmp_t]
            y_avr = (y_c + y_m) / 2
            cpu_numerator.append((x_c - x_avr) * (y_c - y_avr))
            cpu_denominator_1.append((x_c - x_avr) ** 2)
            cpu_denominator_2.append((y_c - y_avr) ** 2)
            mem_numerator.append((x_m - x_avr) * (y_m - y_avr))
            mem_denominator_1.append((x_m - x_avr) ** 2)
            mem_denominator_2.append((y_m - y_avr) ** 2)
        final_cpu = sum(cpu_numerator) / (np.sqrt(sum(cpu_denominator_1)) * np.sqrt(sum(cpu_denominator_2)) + 0.001)
        final_mem = sum(mem_numerator) / (np.sqrt(sum(mem_denominator_1)) * np.sqrt(sum(mem_denominator_2)) + 0.001)
        cor = final_cpu + final_mem
    else:
        for tmp_t in range(t - 5, t + 1):
            x_c = vnf1.trace_cpu[tmp_t]
            x_m = vnf1.trace_mem[tmp_t]
            x_avr = (x_c + x_m) / 2
            y_c = vnf2.trace_cpu[tmp_t]
            y_m = vnf2.trace_mem[tmp_t]
            y_avr = (y_c + y_m) / 2
            cpu_numerator.append((x_c - x_avr) * (y_c - y_avr))
            cpu_denominator_1.append((x_c - x_avr) ** 2)
            cpu_denominator_2.append((y_c - y_avr) ** 2)
            mem_numerator.append((x_m - x_avr) * (y_m - y_avr))
            mem_denominator_1.append((x_m - x_avr) ** 2)
            mem_denominator_2.append((y_m - y_avr) ** 2)
        final_cpu = sum(cpu_numerator) / (np.sqrt(sum(cpu_denominator_1)) * np.sqrt(sum(cpu_denominator_2)) + 0.001)
        final_mem = sum(mem_numerator) / (np.sqrt(sum(mem_denominator_1)) * np.sqrt(sum(mem_denominator_2)) + 0.001)
        cor = final_cpu + final_mem
    return cor


def calculateFit(vnf, node, g: nx.Graph):
    u_c = vnf.req_cpu / g.nodes[node]['cpu_capacity']
    U_c = 1 - g.nodes[node]['cpu_residual'] / g.nodes[node]['cpu_capacity']
    u_m = vnf.req_mem / g.nodes[node]['mem_capacity']
    U_m = 1 - g.nodes[node]['mem_residual'] / g.nodes[node]['mem_capacity']
    l_c = u_c / (u_c + u_m + 0.00001)
    l_m = u_m / (u_c + u_m + 0.00001)
    fit = np.sqrt((l_c - U_c) ** 2) + np.sqrt((l_m - U_m) ** 2)
    return fit
