# EM_BMF

Robust Boolean matrix factorization via EM_BMF
The code is completely process-oriented. Sorry for contaminating your name space.

Dependency: (I think it will work as long as Annaconda on Python3 is installed)

numpy -- 1.11.3

scipy -- 1.1.0

numba -- 0.40.0



Example usage:

```python
def prob_synthesis(shape, latent_size, P, noise_p=0.0):
    '''
    In this synthesis, the probability of X was sampled from the joint probability of the latent factors.
    P is the parameter as Beta(1/(1-p),2) for generating the probability in latent factors.
    '''
    
    a = np.zeros((shape[0], latent_size))
    b = np.zeros((latent_size, shape[1]))
    X = np.ones(shape)
    for l in range(latent_size):
        a[:,l] = np.random.beta(1/(1-P[l]), 2, shape[0])
        b[l,:] = np.random.beta(1/(1-P[l]), 2, shape[1])
        res = X * (1-np.outer(a[:,l],b[l,:]))    
    X = 1 - X
    X_noisy = (1-noise_p)*(X) + noise_p*(1 -X)
    return X_noisy, X, a, b
    
X_noisy, X, a, b = prob_synthesis((1000, 1000), 5, np.random.uniform(0.2,0.5,5), noise_p=0.1)
res1 = EM_BMF(X_noisy, latent_size=5, alpha=0.95, beta=0.95, mask=np.ones(X.shape, dtype=np.int8), max_iter=100)
data = reconstruct(res1[0], res1[1])
print('Reconstruction error:', np.abs(data-X).mean())
```
