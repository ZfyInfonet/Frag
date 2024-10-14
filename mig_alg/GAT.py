import torch
import torch.nn as nn
import numpy as np
import config
import util
import networkx as nx
from datetime import datetime
from torch import Tensor
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv

input_feature_num = 4


class MHGAT(nn.Module):
    def __init__(self, batch_size, g: nx.Graph, out_features_seq, heads_seq, fc_seq, topo_name, device):
        super(MHGAT, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.g = g
        self.edge_index_list = get_edge_index(g, device)
        self.GAT_layers_num = config.gat_layer_num
        self.FCN_layers_num = len(fc_seq)
        self.topo_name = topo_name
        self.model = nn.Sequential()
        self.type = 'MHGAT'
        input_features = [input_feature_num] + [out_features_seq[i] * heads_seq[i] for i in range(len(heads_seq) - 1)]
        fc_input_features = [out_features_seq[-1] * heads_seq[-1]] + [fc_seq[i] for i in range(len(fc_seq))]
        for i in range(len(out_features_seq)):
            self.model.add_module(f'gat{i}', GATConv(in_channels=input_features[i],
                                                     out_channels=out_features_seq[i],
                                                     heads=heads_seq[i],
                                                     add_self_loops=False))
        for i in range(len(fc_seq)):
            self.model.add_module(f'fc{i}', nn.Linear(in_features=fc_input_features[i], out_features=fc_seq[i]))
            self.model.add_module(f'ReLU', nn.ReLU())

    def forward(self, x, e):
        out = torch.zeros((x.shape[0], x.shape[1], 1)).to(self.device)
        for i in range(x.shape[0]):
            xi, ei, yi = x[i], e[i], None
            for layer in range(len(self.model)):
                if layer < self.GAT_layers_num:
                    if layer == 0:
                        yi = self.model[layer](x=xi, edge_index=self.edge_index_list[layer], edge_attr=ei)
                    else:
                        yi = yi + self.model[layer](x=yi, edge_index=self.edge_index_list[layer])
                else:
                    yi = self.model[layer](yi)
            yi.reshape(-1, 1)
            out[i] = yi
        return out

    def net_train(self, dataset, learning_rate, total_epochs):
        model_net_train(self, dataset, learning_rate, total_epochs)


class MHGATwithoutMHG(nn.Module):
    def __init__(self, batch_size, g: nx.Graph, out_features_seq, heads_seq, fc_seq, topo_name, device):
        super(MHGATwithoutMHG, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.g = g
        self.edge_index_list = get_edge_index(g, device)
        self.GAT_layers_num = config.gat_layer_num
        self.FCN_layers_num = len(fc_seq)
        self.topo_name = topo_name
        self.model = nn.Sequential()
        self.type = 'MHGATwithoutMHG'
        input_features = [input_feature_num] + [out_features_seq[i] * heads_seq[i] for i in range(len(heads_seq) - 1)]
        fc_input_features = [out_features_seq[-1] * heads_seq[-1]] + [fc_seq[i] for i in range(len(fc_seq))]
        for i in range(len(out_features_seq)):
            self.model.add_module(f'gat{i}', GATConv(in_channels=input_features[i],
                                                     out_channels=out_features_seq[i],
                                                     heads=heads_seq[i],
                                                     add_self_loops=False))
        for i in range(len(fc_seq)):
            self.model.add_module(f'fc{i}', nn.Linear(in_features=fc_input_features[i], out_features=fc_seq[i]))
            self.model.add_module(f'ReLU', nn.ReLU())

    def forward(self, x, e):
        out = torch.zeros((x.shape[0], x.shape[1], 1)).to(self.device)
        for i in range(x.shape[0]):
            xi, ei, yi = x[i], e[i], None
            for layer in range(len(self.model)):
                if layer < self.GAT_layers_num:
                    if layer == 0:
                        yi = self.model[layer](x=xi, edge_index=self.edge_index_list[0], edge_attr=ei)
                    else:
                        yi = yi + self.model[layer](x=yi, edge_index=self.edge_index_list[0], edge_attr=ei)
                else:
                    yi = self.model[layer](yi)
            yi.reshape(-1, 1)
            out[i] = yi
        return out

    def net_train(self, dataset, learning_rate, total_epochs):
        model_net_train(self, dataset, learning_rate, total_epochs)


class MHGATwithoutResNet(nn.Module):
    def __init__(self, batch_size, g: nx.Graph, out_features_seq, heads_seq, fc_seq, topo_name, device):
        super(MHGATwithoutResNet, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.g = g
        self.edge_index_list = get_edge_index(g, device)
        self.GAT_layers_num = config.gat_layer_num
        self.FCN_layers_num = len(fc_seq)
        self.topo_name = topo_name
        self.model = nn.Sequential()
        self.type = 'MHGATwithoutResNet'
        input_features = [input_feature_num] + [out_features_seq[i] * heads_seq[i] for i in range(len(heads_seq) - 1)]
        fc_input_features = [out_features_seq[-1] * heads_seq[-1]] + [fc_seq[i] for i in range(len(fc_seq))]
        for i in range(len(out_features_seq)):
            self.model.add_module(f'gat{i}', GATConv(in_channels=input_features[i],
                                                     out_channels=out_features_seq[i],
                                                     heads=heads_seq[i],
                                                     add_self_loops=False))
        for i in range(len(fc_seq)):
            self.model.add_module(f'fc{i}', nn.Linear(in_features=fc_input_features[i], out_features=fc_seq[i]))
            self.model.add_module(f'ReLU', nn.ReLU())

    def forward(self, x, e):
        out = torch.zeros((x.shape[0], x.shape[1], 1)).to(self.device)
        for i in range(x.shape[0]):
            xi, ei, yi = x[i], e[i], None
            for layer in range(len(self.model)):
                if layer < self.GAT_layers_num:
                    if layer == 0:
                        yi = self.model[layer](x=xi, edge_index=self.edge_index_list[layer], edge_attr=ei)
                    else:
                        yi = self.model[layer](x=yi, edge_index=self.edge_index_list[layer])
                else:
                    yi = self.model[layer](yi)
            yi.reshape(-1, 1)
            out[i] = yi
        return out

    def net_train(self, dataset, learning_rate, total_epochs):
        model_net_train(self, dataset, learning_rate, total_epochs)


class MHGATwithoutGAT(nn.Module):
    def __init__(self, batch_size, g: nx.Graph, out_features_seq, heads_seq, fc_seq, topo_name, device):
        super(MHGATwithoutGAT, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.g = g
        self.edge_index_list = get_edge_index(g, device)
        self.GAT_layers_num = config.gat_layer_num
        self.FCN_layers_num = len(fc_seq)
        self.topo_name = topo_name
        self.model = nn.Sequential()
        self.type = 'MHGATwithoutGAT'
        input_features = [input_feature_num] + [out_features_seq[i] * heads_seq[i] for i in range(len(heads_seq) - 1)]
        fc_input_features = [out_features_seq[-1] * heads_seq[-1]] + [fc_seq[i] for i in range(len(fc_seq))]
        for i in range(len(out_features_seq)):
            self.model.add_module(f'fc0{i}', nn.Linear(in_features=input_features[i],
                                                       out_features=out_features_seq[i] * heads_seq[i]))
            self.model.add_module(f'ReLU', nn.ReLU())
        for i in range(len(fc_seq)):
            self.model.add_module(f'fc1{i}', nn.Linear(in_features=fc_input_features[i], out_features=fc_seq[i]))
            self.model.add_module(f'ReLU', nn.ReLU())

    def forward(self, x, e):
        out = torch.zeros((x.shape[0], x.shape[1], 1)).to(self.device)
        for i in range(x.shape[0]):
            xi, ei, yi = x[i], e[i], None
            xe = torch.concat([xi.reshape(-1, 1), ei.reshape(-1, 1)]).reshape(-1, input_feature_num)
            xe.to(self.device)
            for layer in range(len(self.model)):
                if layer < self.GAT_layers_num:
                    if layer == 0:
                        yi = self.model[layer](xe)
                    else:
                        yi = yi + self.model[layer](yi)
                else:
                    yi = self.model[layer](yi)
            yi.reshape(-1, 1)
            out[i] = yi[x.shape[1], :]
        return out

    def net_train(self, dataset, learning_rate, total_epochs):
        model_net_train(self, dataset, learning_rate, total_epochs)


class DataSet(Dataset):
    def __getitem__(self, index):
        return self.x[index], self.e[index], self.o[index]

    def __init__(self, x, e, o):
        self.x = x
        self.e = e
        self.o = o

    def __len__(self):
        return self.x.shape[0]


def run(g: nx.Graph, t, overload_nodes, overload_edges, sfc=None, input_model=None):
    mig_cost = 0
    mig_count = 0
    if overload_nodes:
        for src in overload_nodes:
            # remove the VNFs in the overloading nodes
            if mig_count >= config.mig_limit:
                break
            loop_count = 0
            black_list = []
            isOverload = util.isNodeOverload(g, src)
            while isOverload and loop_count < config.loop_limit and mig_count < config.mig_limit:
                loop_count += 1
                vnf = util.getSuitableVnfToMig(g, src, black_list)
                if vnf is None:
                    break
                x, e_attr = util.getLearningInput(g, vnf)
                x = Tensor(np.array([x])).to(input_model.device)
                e_attr = Tensor(np.array([e_attr])).to(input_model.device)
                out = input_model.forward(x, e_attr)
                out = out.reshape(-1).cpu().detach().numpy()
                util_values = out
                util_min = util_values[src - 1]
                dst = None
                for dst_n in g.nodes:
                    if dst_n == src:
                        continue
                    if util.checkNodeResource(vnf, dst_n, g):
                        if util_values[dst_n - 1] < util_min:
                            util_min = util_values[dst_n - 1]
                            dst = dst_n
                if dst is None:
                    black_list.append(vnf)
                    break
                # migrate VNF and links
                mig_result, cost = util.migVnfAndLink(vnf, dst, g)
                if mig_result:
                    mig_cost += cost
                    mig_count += 1
                    isOverload = util.isNodeOverload(g, src)
                else:
                    black_list.append(vnf)

    if overload_edges:
        util.migOverloadEdges(g, overload_edges)

    return mig_count, mig_cost


def model_net_train(model, dataset, learning_rate=5e-4, total_epochs=500):
    print(model.type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_log = np.zeros((total_epochs,))
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_min = 99999
    for epoch in range(0, total_epochs):
        loss_epoch = 0
        train_batch = DataLoader(dataset=dataset, batch_size=model.batch_size, shuffle=True)
        for x, e, o in train_batch:
            x, e, o = x.to(device), e.to(device), o.to(device)
            optimizer.zero_grad()
            net_out = model.forward(x, e)
            loss_function = nn.MSELoss()  # CrossEntropyLoss()
            loss = loss_function(net_out, o)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
        loss_log[epoch] = loss_epoch
        print('Epochs:', epoch + 1, 'Loss =', loss_log[epoch])
        if loss_log[epoch] < loss_min:
            loss_min = int(loss_log[epoch])
    now = datetime.now()
    date_time = now.strftime('%m-%d-%H%M')
    torch.save(model, f=f'./gat_models/{model.topo_name}/{model.type}_loss_{loss_min}_{date_time}')
    torch.save(model, f=f'./gat_models/{model.topo_name}/{model.type}_latest')
    np.save(file=f'./train_logs/{model.type}_{model.topo_name}_log_{date_time}.npy', arr=loss_log)
    np.save(file=f'./train_logs/{model.type}_{model.topo_name}_latest.npy', arr=loss_log)


def get_edge_index(g, device):
    n_num = g.number_of_nodes()
    edge_index_list = []
    adj_mat = g.graph['adj_matrices']
    for layer in range(config.gat_layer_num):
        edge_list = []
        adj_m = adj_mat[layer]
        for i in range(n_num):
            for j in range(i + 1, n_num):
                if adj_m[i, j]:
                    edge_list.append((i, j))
        edge_index = torch.zeros((2, 0), dtype=torch.int).to(device)
        for edge in edge_list:
            tensor_1 = torch.tensor([[edge[0]], [edge[1]]]).to(device)
            tensor_2 = torch.tensor([[edge[1]], [edge[0]]]).to(device)
            edge_index = torch.concat(tensors=(edge_index, tensor_1), dim=1)
            edge_index = torch.concat(tensors=(edge_index, tensor_2), dim=1)
        edge_index_list.append(edge_index)
    return edge_index_list
