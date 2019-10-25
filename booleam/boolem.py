#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 20:03:49 2019

@author: lifan
"""

from numba import jit, prange
import numpy as np
from scipy.special import expit


class boolem:
    def __init__(self, X, latent_size, mask, alpha=0.95, beta=0.95, max_iter=200, em=True, init_eps=0):
        self.X = X
        self.latent_size = latent_size
        self.mask = mask
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.em = em
        self.eps = init_eps
    
    def run(self):
        '''
        Boolean matrix factorization
        '''
        
        w_u = np.random.normal(loc=0, scale=0.1, size=(self.X.shape[0], self.latent_size))
        w_z = np.random.normal(loc=0, scale=0.1, size=(self.latent_size, self.X.shape[1]))
        eps = self.eps
        U = expit(w_u)
        Z = expit(w_z)
        
        if self.em:        
            for i in range(10):
                print('Factor noise:', eps)
                w_u, w_z, loss_trace = self.m_step(w_u, w_z, U, Z, eps)
                U = expit(w_u)
                Z = expit(w_z)
        
                eps_next = self.e_step(U, Z)
                if np.abs(eps_next-eps) < 1e-3: 
                    print('EM stage finished')
                    break
                eps = eps_next
        else:
            w_u, w_z, loss_trace = self.m_step(w_u, w_z, U, Z, eps)
            
        self.U = expit(w_u)
        self.Z = expit(w_z)
        self.eps_hat = eps
        self.loss_trace = loss_trace
        self.X_hat = self.reconstruct(U, Z)

    @staticmethod
    @jit('float64[:,:](float64[:,:], float64[:,:])', nopython=True, nogil=True, parallel=True)
    def reconstruct(U, Z):
        res = np.zeros((U.shape[1],U.shape[0], Z.shape[1]))
        for l in prange(U.shape[1]):
            temp = np.outer(U[:,l], Z[l,:])
            res[l] = np.log(1 - temp)
        return 1 - np.exp(res.sum(0))
    


    @staticmethod
    @jit('Tuple((float64[:,:], float64[:,:]))(int8[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, int8[:,:], float64)', nopython=True, nogil=True, parallel=True)
    def compute_gradient_penalty(X, U, Z, w_u, w_z, alpha, beta, mask, eps):
        '''
        The only difference from the unmasked version is that the gradients computed from masked elements is not taken into account.
        There should be a much more efficint implement, but this is the solution right now for the sake of time.
        '''
            
        x_hat = np.zeros((U.shape[1],U.shape[0], Z.shape[1]))
        for l in prange(U.shape[1]):
            temp = np.outer(U[:,l], Z[l,:])
            x_hat[l] = np.log(1 - temp)
        x_hat = 1 - np.exp(x_hat.sum(0))
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
                grad_wu[i,j] = (temp[j,i,:] * mask[i,:]).sum() + real_alpha * (1 - U[i,j]) - real_beta * U[i,j]
        for i in prange(Z.shape[0]):
            for j in range(Z.shape[1]):
                grad_wz[i,j] = (temp[i,:,j] * mask[:,j]).sum() + real_alpha * (1 - Z[i,j])  - real_beta * Z[i,j]
        return grad_wu, grad_wz

    def e_step(self, U, Z, tol=1e-3):
        x_hat = self.reconstruct(U, Z)
        temp = x_hat>0.5
        return np.abs(self.X*self.mask - temp*self.mask).sum() / self.mask.sum()
    
    def m_step(self, w_u, w_z, U, Z, eps, initial_lr=0.1, lr_min=1e-6, lr_max=1.0, plus_factor=1.2, minus_factor=0.5):
        grad_wu_prev, grad_wz_prev = np.zeros(w_u.shape), np.zeros(w_z.shape)
        lr_wu, lr_wz = np.ones(w_u.shape) * initial_lr, np.ones(w_z.shape) * initial_lr
        change_wu, change_wz = np.zeros(w_u.shape), np.zeros(w_z.shape)
        loss_trace = []
        U_prev = U > 0.5
        Z_prev = Z > 0.5
        
        for i in range(self.max_iter):
            grad_wu, grad_wz = self.compute_gradient_penalty(self.X, U, Z, w_u, w_z, self.alpha, self.beta, self.mask, eps)
            lr_wu, grad_wu_prev, change_wu = self.lr_update(grad_wu_prev, grad_wu, change_wu, lr_wu, lr_min, lr_max, plus_factor, minus_factor)
            w_u = w_u + change_wu
            lr_wz, grad_wz_prev, change_wz = self.lr_update(grad_wz_prev, grad_wz, change_wz, lr_wz, lr_min, lr_max, plus_factor, minus_factor)
            w_z = w_z + change_wz
            
            w_u = self.clip(w_u,5)
            w_z = self.clip(w_z,5)
            
            U = expit(w_u)
            Z = expit(w_z)
    
            if (i+1) % 10 == 0:
                temp = self.reconstruct(U, Z)
                temp = (1-eps) * temp + eps * (1-temp)
                loss = - self.X * np.log(temp) - (1-self.X) * np.log(1-temp)
                loss_trace.append(loss.sum())
                print('iteration:', i+1,'--loss:', loss_trace[-1], end='\n')
                U_bool = U > 0.5
                Z_bool = Z > 0.5
                if np.all(U_prev==U_bool) & np.all(Z_prev==Z_bool): break
                U_prev, Z_prev = U_bool, Z_bool
            
        return w_u, w_z, loss_trace

    @staticmethod
    @jit('Tuple((float64[:,:], float64[:,:], float64[:,:]))(float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64, float64, float64, float64)', nopython=True, nogil=True, parallel=True)
    def lr_update(grad_prev, grad, change, lr, lr_min=1e-6, lr_max=1.0, plus_factor=1.2, minus_factor=0.5):
        for i in range(grad.shape[0]):
            for j in prange(grad.shape[1]):
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
    
    def clip(self, m, thre):
        m[m>thre] = thre
        m[m<-thre] = -thre
        return m