## Question Link
https://www.deep-ml.com/problems/37

## My Solution
```
import numpy as np
def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        Y=X
    nX = X.shape[1]
    nY = Y.shape[1]
    XY = np.hstack((X,Y))
    covariance = np.cov(XY,rowvar=False)
    cov_XY = covariance[:nX, nX:]
    sigma_x = np.std(X,axis=0,ddof=1)
    sigma_y = np.std(Y,axis=0,ddof=1)
    output = cov_XY/np.outer(sigma_x,sigma_y)
    return output
```

* hstack to stack columns next to each other