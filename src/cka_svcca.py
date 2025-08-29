\
import torch
import numpy as np
from sklearn.cross_decomposition import CCA

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    Kx = X @ X.T
    Ky = Y @ Y.T
    hsic = (Kx*Ky).mean() - Kx.mean()*Ky.mean() - (Kx.mean(0)*Ky.mean(0)).mean() + Kx.mean()*Ky.mean()
    varx = (Kx*Kx).mean() - 2*Kx.mean()*Kx.mean(0).mean() + Kx.mean()**2
    vary = (Ky*Ky).mean() - 2*Ky.mean()*Ky.mean(0).mean() + Ky.mean()**2
    return (hsic / (torch.sqrt(varx*vary) + 1e-8)).item()

def svcca(X: torch.Tensor, Y: torch.Tensor, n_comp: int=64):
    cca = CCA(n_components=min(n_comp, X.shape[1], Y.shape[1]))
    U, V = cca.fit_transform(X.numpy(), Y.numpy())
    corr = np.corrcoef(U.T, V.T).diagonal(offset=U.shape[1]).mean()
    return float(corr)
