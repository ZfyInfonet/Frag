import init
import torch
from util import *
from datetime import datetime
init_time = datetime.now()

# 0: NSFNET   1: USbackbone
topo_index = 0
dataset_index = 0
topo_name = 'USbackbone' if topo_index else 'NSFNET'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = ['MHGAT', 'MHGATwithoutMHG', 'MHGATwithoutResNet', 'MHGATwithoutGAT']
models = []
for model in model_name:
    load_model = torch.load(f'./gat_models/{topo_name}/{model}_latest')
    models.append(load_model)
x = torch.load(f'./dataset/GAT_test_data/{topo_name}/x.pt').to(device)
e = torch.load(f'./dataset/GAT_test_data/{topo_name}/e.pt').to(device)
o = torch.load(f'./dataset/GAT_test_data/{topo_name}/o.pt').to(device)
result = np.zeros([x.shape[0], len(models) + 1])
print(x.shape)
for i in range(x.shape[0]):
    xi, ei = x[i].reshape([1, x.shape[1], x.shape[2]]), e[i].reshape([1, e.shape[1], e.shape[2]])
    oi = o[i].reshape([1, o.shape[1], o.shape[2]])
    for j in range(len(models)):
        result[i, 0] = i
        output = models[j].forward(xi, ei)
        loss_function = torch.nn.MSELoss()
        loss = loss_function(output, oi)
        result[i, j + 1] = loss.item()

mean = np.mean(result, axis=0)[1:]
var = np.var(result, axis=0)[1:]
print(mean, '\n', var)
df = pd.DataFrame(result)
df.to_excel(f'./result/net_sim/{topo_name}.xlsx', index=False, header=False)
