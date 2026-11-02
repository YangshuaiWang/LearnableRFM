# LearnableRFM

One-dimensional random feature models accompanying [*LDD-RFM: Learnable Domain Decomposition for Random Feature Models via Variable Projection*](https://openreview.net/forum?id=OzYGQoPGki) (NeurIPS 2026).

The examples in [`ldd_rfm/models.py`](ldd_rfm/models.py) fit a scalar function using three choices of partition:

| Model | Partition and fit |
| --- | --- |
| `GlobalRFM` | A single global random feature basis with ridge regression. |
| `FixedPoURFM` | Compact cosine windows on `[-1, 1]`, defined in [`ldd_rfm/pou.py`](ldd_rfm/pou.py), weight the local feature bases. |
| `LDDRFM` | A softmax network learns the partition. The model alternates a joint ridge solve for the feature coefficients with Adam steps for the gate; the random features and their spectral scale stay fixed. |

## Example

The target below is the piecewise sine function from Appendix C.1 of the paper. NumPy is sufficient for the first two models; the learned gate also needs PyTorch. From a checkout of this repository, install the dependencies with `python -m pip install numpy torch`, then run:

```python
import numpy as np
from ldd_rfm.models import LDDRFM

x = np.linspace(-1.0, 1.0, 256)
y = np.where(x <= 0, np.sin(2 * x), np.sin(20 * x))

model = LDDRFM(M=2, K_per=32, outer_iters=10, seed=0)
model.fit(x[:, None], y)
prediction = model.predict(np.array([[-0.75], [0.25]]))
```

This repository illustrates the one-dimensional regression models. The PDE benchmarks and experiment scripts are outside its scope.
