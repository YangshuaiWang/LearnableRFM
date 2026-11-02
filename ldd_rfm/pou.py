import numpy as np

def build_pou_1d(X, M):
    x = X[:,0]
    centers = np.linspace(-1+1/M, 1-1/M, M)
    radius = 2.0/M
    W = np.zeros((len(x), M))
    for m,c in enumerate(centers):
        dist = np.abs(x-c)
        w = np.zeros_like(x)
        mask = dist <= radius
        w[mask] = 0.5*(1.0 + np.cos(np.pi*dist[mask]/radius))
        W[:,m] = w
    s = W.sum(axis=1, keepdims=True)+1e-12
    return W/s, centers, radius