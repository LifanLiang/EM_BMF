from numba import jit, prange
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import pandas as pd

def EM_BMF(X, latent_size, mask, alpha = 0.95, beta = 0.95, max_iter=200, em=True):
    '''
    Boolean matrix factorization by using an identity matrix as input and observation as output for a neural network.
    The edge weights learned through the network can then be interpreted as the factor matrixes.
    The lesser size of x will be used as the output.
    '''
    
    w_u = np.random.normal(loc=0, scale=0.1, size=(X.shape[0], latent_size))
    w_z = np.random.normal(loc=0, scale=0.1, size=(latent_size, X.shape[1]))
    eps = 0.0
    U = expit(w_u)
    Z = expit(w_z)
    
    for i in range(10):
        print('Factor noise:', eps)
        w_u, w_z, loss_trace = m_step1(X, w_u, w_z, U, Z, alpha, beta, mask, eps, max_iter)
        U = expit(w_u)
        Z = expit(w_z)

        eps_next = e_step(X, U, Z, mask)
        if np.abs(eps_next-eps) < 1e-3: 
            print('EM stage finished')
            break
        eps = eps_next
    return expit(w_u), expit(w_z), loss_trace

def e_step(X, U, Z, mask, tol=1e-3):
    x_hat = reconstruct(U, Z)
    temp = x_hat>0.5
    return np.abs(X*mask - temp*mask).sum() / mask.sum()


def m_step1(X, w_u, w_z, U, Z, alpha, beta, mask, eps, max_iter, initial_lr=0.1, lr_min=1e-6, lr_max=1.0, plus_factor=1.2, minus_factor=0.5):
    grad_wu_prev, grad_wz_prev = np.zeros(w_u.shape), np.zeros(w_z.shape)
    lr_wu, lr_wz = np.ones(w_u.shape) * initial_lr, np.ones(w_z.shape) * initial_lr
    change_wu, change_wz = np.zeros(w_u.shape), np.zeros(w_z.shape)
    loss_trace = []
    U_prev = U > 0.5
    Z_prev = Z > 0.5
    
    for i in range(max_iter):
        grad_wu, grad_wz = compute_gradient_penalty(X, U, Z, w_u, w_z, alpha, beta, mask, eps)
        lr_wu, grad_wu_prev, change_wu = lr_update(grad_wu_prev, grad_wu, change_wu, lr_wu, lr_min, lr_max, plus_factor, minus_factor)
        w_u = w_u + change_wu
        lr_wz, grad_wz_prev, change_wz = lr_update(grad_wz_prev, grad_wz, change_wz, lr_wz, lr_min, lr_max, plus_factor, minus_factor)
        w_z = w_z + change_wz
        
        w_u = clip(w_u,5)
        w_z = clip(w_z,5)
        
        U = expit(w_u)
        Z = expit(w_z)

        if (i+1) % 10 == 0:
            temp = reconstruct(U, Z)
            temp = (1-eps) * temp + eps * (1-temp)
            loss = - X * np.log(temp) - (1-X) * np.log(1-temp)
            loss_trace.append(loss.sum())
            print('iteration:', i+1,'--loss:', loss_trace[-1], end='\n')
            U_bool = U > 0.5
            Z_bool = Z > 0.5
            if np.all(U_prev==U_bool) & np.all(Z_prev==Z_bool): break
            U_prev, Z_prev = U_bool, Z_bool
        
    return w_u, w_z, loss_trace
        
    
@jit('Tuple((float64[:,:], float64[:,:], float64[:,:]))(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, float64)', nopython=True, nogil=True, parallel=True)
def lr_update(grad_prev, grad, change, lr, lr_min=1e-6, lr_max=1.0, plus_factor=1.2, minus_factor=0.5):
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            sign = grad_prev[i,j] * grad[i,j]
            if sign > 0:
                lr[i,j] = min((lr[i,j] * plus_factor, lr_max))
                change[i,j] = -np.sign(grad[i,j]) * lr[i,j]
            elif sign < 0:
                lr[i,j] = max((lr[i,j] * minus_factor, lr_min))
                change[i,j] = -change[i,j]
                grad[i,j] = 0
            elif sign == 0:
                change[i,j] = -np.sign(grad[i,j]) * lr[i,j]
    return lr, grad, change

@jit('float64[:,:](float64[:,:], float64[:,:])', nopython=True, nogil=True, parallel=True)
def reconstruct(U, Z):
    res = np.zeros((U.shape[1],U.shape[0], Z.shape[1]))
    for l in prange(U.shape[1]):
        temp = np.outer(U[:,l], Z[l,:])
        res[l] = np.log(1 - temp)
    return 1 - np.exp(res.sum(0))

@jit('Tuple((float64[:,:], float64[:,:]))(int8[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, int8[:,:], float64)', nopython=True, nogil=True, parallel=True)
def compute_gradient_penalty(X, U, Z, w_u, w_z, alpha, beta, mask, eps):
    '''
    The only difference from the unmasked version is that the gradients computed from masked elements is not taken into account.
    There should be a much more efficint implement, but this is the solution right now for the sake of time.
    '''
    
    x_hat = reconstruct(U, Z)
    x_noisy = (1-eps) * x_hat + eps * (1-x_hat)
    temp = np.zeros((U.shape[1], U.shape[0], Z.shape[1]), dtype=np.float64)
    temp1 = (1 - X / x_noisy) * (1 - x_hat) / (1 - x_noisy)
    real_alpha = alpha - 1
    real_beta = beta - 1
    for l in prange(U.shape[1]):
        temp_uz = np.outer(U[:,l], Z[l,:])
        temp[l] = temp_uz / (1 - temp_uz)
    for l in range(U.shape[1]):
        temp[l] = temp[l] * temp1

    grad_wu = np.empty((U.shape))
    grad_wz = np.empty((Z.shape))

    for i in prange(U.shape[0]):
        for j in range(U.shape[1]):
            grad_wu[i,j] = (temp[j,i,:] * mask[i,:]).sum() + real_alpha / U[i,j] - real_beta / (1 - U[i,j])
    for i in prange(Z.shape[0]):
        for j in range(Z.shape[1]):
            grad_wz[i,j] = (temp[i,:,j] * mask[:,j]).sum() + real_alpha / Z[i,j] - real_beta / (1 - Z[i,j])
    return grad_wu, grad_wz

def clip(m, thre):
    m[m>thre] = thre
    m[m<-thre] = -thre
    return m
