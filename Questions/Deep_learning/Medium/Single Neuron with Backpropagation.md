## Question Link
https://www.deep-ml.com/problems/25

## My Solution
```
import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    weights = initial_weights
    bias = initial_bias
    mse_values=[]
	for i in range(epochs):
        preds = np.dot(features,weights) + bias
        out = 1/(1+np.exp(-preds))
        loss = np.mean((out-labels)**2)
        mse_values.append(round(loss,4))

        d_loss = ((out-labels)*2)/features.shape[0]
        d_sigmoid = out*(1-out)
        d_out = d_loss*d_sigmoid

        d_w = np.dot(features.T,d_out)
        d_b = np.sum(d_out)

        weights -= learning_rate*d_w
        bias -= learning_rate*d_b
    updated_weights = weights
    updated_bias = bias
    return updated_weights, updated_bias, mse_values
```

* Just keep chain rule in mind