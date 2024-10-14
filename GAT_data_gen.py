import copy

import config
import init
import deploy
import util
import torch
from util import *
from datetime import datetime


def generate_data(is_test=1, topo_name=None, data_number=5000):
    data_num = data_number
    data_index = 0
    output_directory = 'GAT_test_data' if is_test else 'GAT_data'
    n_num, e_num = init.getTopoNodeAndEdgeNum(topo_name)
    print(n_num+e_num)
    arrive_rate = 10
    Time = config.Time
    data_count = 0
    x_tensor = np.zeros((data_num, n_num, 4))
    e_tensor = np.zeros((data_num, e_num * 2, 2))
    o_tensor = np.zeros((data_num, n_num, 1))
    g_init = init.getGraph('node20')
    while data_count < data_num:
        # initialization
        for arr_rate in range(1, arrive_rate + 1):
            if data_count == data_num:
                break
            sfc_num = arr_rate * Time
            g = copy.deepcopy(g_init)
            sfc_list = init.getSfcSet(sfc_num, g.number_of_nodes(), data_index)
            sfc_arrive_order = init.getArrivalOrder(sfc_num, arr_rate)
            sfc_list_cp = copy.deepcopy(sfc_list)
            sfc_id = 0
            arrived_sfc_count = 0
            for t in range(Time):
                if data_count == data_num:
                    break
                # update resource utilization
                over_n, over_e, _, _, _, frag = updateGraphAndServices(g, sfc_list_cp, t)

                # print('Now:')
                if over_n:
                    for node in over_n:
                        if_skip = False
                        black_list = []
                        vnf = getSuitableVnfToMig(g, node, black_list)
                        x, e_attr = util.getLearningInput(g, vnf)
                        out = np.zeros((n_num, 1))

                        for dst in g.nodes:
                            graph = copy.deepcopy(g)
                            new_vnf = None
                            for v in graph.nodes[vnf.node_id]['vnf_list']:
                                if v.sfc_id == vnf.sfc_id and v.vnf_id == vnf.vnf_id:
                                    new_vnf = v
                                    break

                            # migrate VNF and links
                            mig_result, cost = util.migVnfAndLink(new_vnf, dst, graph)
                            frag = util.getGraphFrag(graph, config.receptive_field)
                            out[dst - 1, 0] = max(frag)
                        if not if_skip:
                            x_tensor[data_count] = x
                            e_tensor[data_count] = e_attr
                            o_tensor[data_count] = out

                            data_count += 1
                            print(data_count)
                            if data_count == data_num:
                                break
                if data_count == data_num:
                    break
                # sfc arriving
                if t < len(sfc_arrive_order):
                    # for each sfc arrived at time t
                    for j in range(sfc_arrive_order[t]):
                        sfc = sfc_list_cp[sfc_id + j]
                        sfc.arrive_time = t
                        arrived_sfc_count += 1
                        _ = deploy.run(sfc, g)

                    sfc_id += sfc_arrive_order[t]

    x = torch.Tensor(x_tensor)
    e = torch.Tensor(e_tensor)
    o = torch.Tensor(o_tensor)
    torch.save(x, f'./dataset/{output_directory}/{topo_name}/x.pt')
    torch.save(e, f'./dataset/{output_directory}/{topo_name}/e.pt')
    torch.save(o, f'./dataset/{output_directory}/{topo_name}/o.pt')


generate_data(is_test=0, topo_name='node10', data_number=5000)
