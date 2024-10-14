import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau


def covariance(x, y):
    n = len(x)
    # 计算x和y的平均值
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    # 计算协方差
    return np.sum((x - x_bar) * (y - y_bar)) / (n - 1)


result = np.zeros((6, 4))
data = pd.read_excel('data1000_origin.xlsx').to_numpy().T
[x, y1, y2, y3, y4, y5, y6] = data
metrics = [y1, y2, y3, y4, y5, y6]
for i in range(6):
    print(f' metrics {i}:')
    pearsonr_v, _ = pearsonr(x, metrics[i])
    print(f' pearsonr: {pearsonr_v}')
    spearmanr_v, _ = spearmanr(x, metrics[i])
    print(f' spearmanr: {spearmanr_v}')
    kendalltau_v, _ = kendalltau(x, metrics[i])
    print(f' kendalltau: {kendalltau_v}')
    cov_v = covariance(x, metrics[i])
    print(f' cov: {cov_v}')
    result[i] = np.array([cov_v, pearsonr_v, spearmanr_v, kendalltau_v])
print(result)
df = pd.DataFrame(result)
df.to_excel(f'corr_comparison.xlsx', index=False, header=False)
