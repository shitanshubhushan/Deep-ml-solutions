## Question Link
https://www.deep-ml.com/problems/131

## My Solution
```
import numpy as np
def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    seq_len,d_k = Q.shape
    _,d_v = V.shape
    out = np.zeros((seq_len, d_v))

    if scale_factor is None:
        scale_factor = np.sqrt(d_k)
    
    for i in range(seq_len):
        left = max(0,i-window_size)
        right = min(seq_len-1,i+window_size)
        window_indices = np.arange(left,right+1)

        q = Q[i]                
        k_window = K[window_indices]  
        v_window = V[window_indices]   
        attn_scores = np.dot(k_window, q) / scale_factor   

        attn_weights = np.exp(attn_scores - np.max(attn_scores))
        attn_weights = attn_weights / np.sum(attn_weights)

        out[i] = np.dot(attn_weights, v_window)
    
    return out
```

* Need to practice this multiple times