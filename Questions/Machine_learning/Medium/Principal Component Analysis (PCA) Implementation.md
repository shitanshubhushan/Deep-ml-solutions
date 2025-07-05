## Question Link:
https://www.deep-ml.com/problems/19

## My Solution:
```
import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    data_std = (data-mean)/std_dev

    rows = data_std.shape[0]
    col = data_std.shape[1]

    cov = np.cov(data_std, rowvar=False)

    eigvals, eigvecs = np.linalg.eig(cov)

    sorted_idx = eigvals.argsort()[::-1]
    sorted_eigvals = eigvals[sorted_idx]
    sorted_eigvecs = eigvecs[:,sorted_idx]
    principal_components=sorted_eigvecs[:,:k]
    return np.round(principal_components, 4)
```

**Key Steps**
Standardise -> covariance -> eigen vals and vectors -> sort

* For Covariance keep rowvar=False