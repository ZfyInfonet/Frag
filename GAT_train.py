import torch
import init
from mig_alg.GAT import MHGAT, MHGATwithoutMHG, MHGATwithoutResNet, MHGATwithoutGAT, DataSet


model_index = 0
topo_index = 0      # 0: NSFNET   1: USbackbone
dataset_index = 0
topo_name = 'node10'
model_name = [MHGAT, MHGATwithoutMHG, MHGATwithoutResNet, MHGATwithoutGAT]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
graph = init.getGraph(topo_name)
x = torch.load(f'./dataset/GAT_data/{topo_name}/x.pt').to(device)
e = torch.load(f'./dataset/GAT_data/{topo_name}/e.pt').to(device)
o = torch.load(f'./dataset/GAT_data/{topo_name}/o.pt').to(device)
train_data = DataSet(x, e, o)
model = model_name[model_index]
model_to_train = model(batch_size=64,
                       g=graph,
                       out_features_seq=[256, 256, 256],
                       heads_seq=[2, 2, 2],
                       fc_seq=[256, 256, 1],
                       topo_name=topo_name,
                       device=device
                       ).to(device=device)
# train
model_to_train.net_train(train_data, learning_rate=5e-4, total_epochs=50)

# normalization
# instance_norm = nn.InstanceNorm1d(num_features=2)
# x, e, o = x.transpose(1, 2), e.transpose(1, 2), o.transpose(1, 2)
# x, e, o = instance_norm(x), instance_norm(e), instance_norm(o)
# x, e, o = x.transpose(1, 2), e.transpose(1, 2), o.transpose(1, 2)

# create net
# x_cpu, x_mem, e_bw, e_delay, o_frag, o_cost = x[:, :, 0], x[:, :, 1], e[:, :, 0], e[:, :, 1], o[:, :, 0], o[:, :, 1]
# w_cpu, w_mem = torch.amax(x_cpu), torch.amax(x_mem)
# w_bw, w_delay = torch.amax(e_bw), torch.amax(e_delay)
# w_frag, w_cost = torch.amax(o_frag), torch.amax(o_cost)
# if w_cpu.numpy() != 0:
#     x_cpu /= w_cpu
# if w_mem.numpy() != 0:
#     x_cpu /= w_mem
# if w_bw.numpy() != 0:
#     e_bw /= w_bw
# if w_delay.numpy() != 0:
#     e_delay /= w_delay
# if w_frag.numpy() != 0:
#     o_frag /= w_frag
# if w_cost.numpy() != 0:
#     o_cost /= w_cost
