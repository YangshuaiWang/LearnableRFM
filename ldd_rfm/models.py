import numpy as np
from .pou import build_pou_1d

# ---- utilities ----
def _features(X, K, R, activation, seed):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.uniform(-R, R, size=(K,d))
    b = rng.uniform(-R, R, size=(K,))
    Z = X @ W.T + b
    Phi = np.tanh(Z) if activation=="tanh" else np.cos(Z)
    return Phi, W, b

def _ridge(Phi, y, lam):
    K = Phi.shape[1]
    A = Phi.T @ Phi + lam*np.eye(K)
    b = Phi.T @ y
    return np.linalg.solve(A, b)

# ---- Baseline: Global RFM ----
class GlobalRFM:
    def __init__(self, K=1024, R=8.0, activation="tanh", lam=1e-5, seed=0):
        self.K,self.R,self.activation,self.lam,self.seed = K,R,activation,lam,seed
        self.W_=None; self.b_=None; self.w_=None
    def fit(self, X, y):
        Phi, W, b = _features(X, self.K, self.R, self.activation, self.seed)
        w = _ridge(Phi, y, self.lam)
        self.W_, self.b_, self.w_ = W, b, w
        return self
    def predict(self, X):
        Z = X @ self.W_.T + self.b_
        Phi = np.tanh(Z) if self.activation=="tanh" else np.cos(Z)
        return Phi @ self.w_

# ---- Baseline: Fixed-PoU RFM ----
class FixedPoURFM:
    def __init__(self, M=4, K_per=256, R=8.0, activation="tanh", lam=1e-5, seed=0):
        self.M,self.K_per,self.R,self.activation,self.lam,self.seed = M,K_per,R,activation,lam,seed
        self.params_=[]; self.M_ = M
    def fit(self, X, y):
        Wpou,_,_ = build_pou_1d(X, self.M_)
        rng = np.random.default_rng(self.seed)
        for m in range(self.M_):
            Phi_m, W_m, b_m = _features(X, self.K_per, self.R, self.activation, rng.integers(1<<30))
            Phi_w = (Wpou[:,m:m+1]) * Phi_m
            w_m = _ridge(Phi_w, y, self.lam)
            self.params_.append((W_m,b_m,w_m))
        self.Wpou_ = Wpou
        return self
    def predict(self, X):
        Wpou,_,_ = build_pou_1d(X, self.M_)
        pred = np.zeros(X.shape[0])
        for m,(W_m,b_m,w_m) in enumerate(self.params_):
            Z = X @ W_m.T + b_m
            Phi_m = np.tanh(Z) if self.activation=="tanh" else np.cos(Z)
            pred += Wpou[:,m] * (Phi_m @ w_m)
        return pred

# ---- LDD-RFM (optional; requires torch for learnable gates) ----
try:
    import torch
    class GateNet(torch.nn.Module):
        def __init__(self, d, M, hidden=64, tau=1.0):
            super().__init__()
            self.fc1 = torch.nn.Linear(d, hidden)
            self.fc2 = torch.nn.Linear(hidden, M)
            self.register_buffer("tau", torch.tensor(float(tau)))
        def forward(self, x):
            z = torch.tanh(self.fc1(x))
            scores = self.fc2(z)
            return torch.softmax(scores / self.tau, dim=1)

    class LDDRFM:
        def __init__(self, M=4, K_per=256, R=8.0, activation="tanh", lam=1e-5,
                     gate_hidden=64, tau0=1.0, tau_min=0.1, tau_decay=0.95,
                     gate_steps=1, gate_lr=1e-2, outer_iters=20, seed=0):
            self.M=M; self.K_per=K_per; self.R=R; self.activation=activation; self.lam=lam
            self.gate_hidden=gate_hidden; self.tau0=tau0; self.tau_min=tau_min; self.tau_decay=tau_decay
            self.gate_steps=gate_steps; self.gate_lr=gate_lr; self.outer_iters=outer_iters; self.seed=seed
            self.feature_params_=None; self.w_concat_=None; self.gate_=None

        def _phi(self, X, W, b):
            Z = X @ W.T + b
            return np.tanh(Z) if self.activation=="tanh" else np.cos(Z)

        def fit(self, X, y, verbose=False):
            rng = np.random.default_rng(self.seed)
            n, d = X.shape
            # random features per expert
            self.feature_params_ = []
            for _ in range(self.M):
                W_m = rng.uniform(-self.R, self.R, size=(self.K_per, d))
                b_m = rng.uniform(-self.R, self.R, size=(self.K_per, ))
                self.feature_params_.append((W_m, b_m))
            torch.manual_seed(self.seed)
            self.gate_ = GateNet(d, self.M, hidden=self.gate_hidden, tau=self.tau0)
            opt = torch.optim.Adam(self.gate_.parameters(), lr=self.gate_lr)
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y.reshape(-1,1), dtype=torch.float32)
            tau = self.tau0
            for it in range(self.outer_iters):
                with torch.no_grad():
                    A = self.gate_(Xt).cpu().numpy()   # (n,M)
                blocks=[]
                for m in range(self.M):
                    W_m,b_m = self.feature_params_[m]
                    Phi_m = self._phi(X, W_m, b_m)          # (n,K)
                    Phi_w = A[:,m:m+1] * Phi_m
                    blocks.append(Phi_w)
                Xbig = np.concatenate(blocks, axis=1)       # (n, M*K)
                Ktot = self.M*self.K_per
                A_mat = Xbig.T @ Xbig + self.lam*np.eye(Ktot)
                b_vec = Xbig.T @ y
                self.w_concat_ = np.linalg.solve(A_mat, b_vec)

                for _ in range(self.gate_steps):
                    opt.zero_grad()
                    A_t = self.gate_(Xt)                    # (n,M)
                    preds=[]
                    start=0
                    for m in range(self.M):
                        W_m,b_m = self.feature_params_[m]
                        Phi_m = self._phi(X, W_m, b_m)
                        Phi_m_t = torch.tensor(Phi_m, dtype=torch.float32)
                        w_m = self.w_concat_[start:start+self.K_per]
                        w_m_t = torch.tensor(w_m, dtype=torch.float32).unsqueeze(1)
                        pred_m = (Phi_m_t @ w_m_t).squeeze(1)
                        preds.append(A_t[:,m]*pred_m)
                        start += self.K_per
                    pred = torch.stack(preds, dim=1).sum(dim=1, keepdim=True)
                    loss = torch.mean((pred-yt)**2)
                    loss.backward()
                    opt.step()
                # anneal tau
                tau = max(self.tau_min, tau*self.tau_decay)
                self.gate_.tau = torch.tensor(float(tau))
                if verbose and (it%5==0 or it==self.outer_iters-1):
                    with torch.no_grad():
                        A_t = self.gate_(Xt)
                        preds=[]
                        start=0
                        for m in range(self.M):
                            W_m,b_m = self.feature_params_[m]
                            Phi_m = self._phi(X, W_m, b_m)
                            Phi_m_t = torch.tensor(Phi_m, dtype=torch.float32)
                            w_m = self.w_concat_[start:start+self.K_per]
                            w_m_t = torch.tensor(w_m, dtype=torch.float32).unsqueeze(1)
                            pred_m = (Phi_m_t @ w_m_t).squeeze(1)
                            preds.append(A_t[:,m]*pred_m)
                            start += self.K_per
                        pred = torch.stack(preds, dim=1).sum(dim=1, keepdim=True)
                        print(f"[LDD-RFM] iter {it:03d} mse={torch.mean((pred-yt)**2).item():.3e} tau={tau:.3f}")
            return self

        def predict(self, X):
            Xt = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                A = self.gate_(Xt).cpu().numpy()
            preds=[]
            start=0
            for m in range(self.M):
                W_m,b_m = self.feature_params_[m]
                Phi_m = self._phi(X, W_m, b_m)
                w_m = self.w_concat_[start:start+self.K_per]
                preds.append(A[:,m] * (Phi_m @ w_m))
                start += self.K_per
            return np.sum(np.stack(preds, axis=1), axis=1)

except Exception:
    # Torch not available: define a stub so imports won't fail
    class LDDRFM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available; LDD-RFM gating requires torch.")