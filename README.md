# Installing EM_BMF (BEM)

Use pip install. You can also visit https://pypi.org/project/boolem/0.0.5/

```
pip install boolem==0.0.5
```


# Dependency:

(I think it will work as long as Annaconda on Python3 is installed)

numpy -- 1.11.3

scipy -- 1.1.0

numba -- 0.40.0

# Example usage:

```python
import numpy as np
from boolem import BEM

def synthesis(shape, latent_size, P, noise_p=0.0):
    '''
    In this synthesis, the probability of X was sampled from the joint probability of the latent factors.
    P is the parameter as Beta(1/(1-p),2) for generating the probability in latent factors.
    '''
    
    a = np.zeros((shape[0], latent_size))
    b = np.zeros((latent_size, shape[1]))
    X = np.zeros(shape)
    for l in range(latent_size):
        a[:,l] = np.random.binomial(1, P[l], shape[0])
        b[l,:] = np.random.binomial(1, P[l], shape[1])
        X += np.outer(a[:,l],b[l,:]) 
    X[X>1] = 1
    flip = np.random.binomial(1, noise_p, X.shape)
    X_noisy = np.abs(X-flip)
    return X_noisy, X, a, b

# Generate a Boolean matrice with heterogeneous Boolean factors and uniform noise.   
X_noisy, X, a, b = synthesis((1000, 1000), 4, np.random.uniform(0.2,0.5,4), noise_p=0.2)

# Feed the model with noisy matrix. 
# Latent_size: the dimension of latent Boolean factors. 
# alpha: the alpha for the beta prior. Default is recommended.
# beta: the beta for the beta prior. Default is recommended.
# mask: the matrix with the same shape as X. 0 means the correponding element in X is missing.
# max_iter: the maximum iteration for gradient-based optimization
model = BEM(np.int8(X_noisy), latent_size=4, alpha=0.95, beta=0.95, mask=np.ones(X.shape, dtype=np.int8), max_iter=200)
model.run()

# After running factorization, the model will contain several new attributes as the output:
# model.U: the latent factor with the shape (X.shape[0], latent_size)
# model.Z: the latent facotr with the shape (latent_size, X.shape[1])
# model.X_hat: reconstructed Boolean matrix from U and Z. Note that values in X_hat is continuous within [0,1]
print('Reconstruction error:', np.abs((model.X_hat>0.5)-X).mean())

# You can also try to find the right number of factors by searching for the minimum AIC
model.AIC

```
