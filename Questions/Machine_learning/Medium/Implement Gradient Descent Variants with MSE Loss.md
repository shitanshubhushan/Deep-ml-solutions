## Question Link:
https://www.deep-ml.com/problems/47

## My Solution:
```
import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    n = X.shape[0]
    if method == 'batch':
        for i in range(0,n_iterations):
            pred = np.dot(X,weights)
            grad = (2/n) * np.dot(X.T, (pred - y))
            weights = weights - learning_rate*grad
    if method == 'stochastic':
        for i in range(0,n_iterations):
            for x_data,y_data in zip(X,y):
                pred = np.dot(x_data,weights)
                grad = 2*(pred-y_data)*x_data
                weights = weights - learning_rate*grad
    if method == 'mini_batch':
        for i in range(0,n_iterations):
            for start in range(0,n,batch_size):
                end = min(start+batch_size,n)
                x_data = X[start:end]
                y_data = y[start:end]
                pred = np.dot(x_data,weights)
                grad = (2/batch_size) * np.dot(x_data.T, (pred - y_data))
                weights = weights - learning_rate*grad
    return weights
```

* batch does all samples in 1 go
* stochastic does seperately for each sample
