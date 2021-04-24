#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from numpy import random as rd

def generar_lineas(n_lines = 100000, N = 5, y0 = 0, yN = 1):

    w = rd.normal(scale = 1 / np.sqrt(2),size = (n_lines,N-1))
    w = np.hstack([np.array([[y0]*n_lines]).T, w, np.array([[yN]*n_lines]).T])

    z = w[:]
    
    for i in range(1,N):
        z[:,i] = w[:,i] * np.sqrt(i / (i+1)) * np.sqrt(4 / N)
        
    else:
        y = z[:]
        
    for i in range(1,N)[::-1]:
        y[:,i] = z[:,i] + y0 / (i + 1) + (i / (i+1))*y[:,i+1]
        
    return y

def return_accion(S,delta,N):
    def return_accion_(S,delta,N):
        return ((S - delta**2 /4 )**(-1 + (N-1) / 2)) * np.exp(-(S - delta**2 / 4)) / gamma((N-1) / 2)

    to_return  =  (S > delta**2 /4 )
    no_return  =  ~to_return

    return np.hstack([np.zeros(no_return.sum()),return_accion_(S[to_return], delta, N)])

def generar_accion(y):
    
    N = y.shape[1] - 1
    accion = (y[:,1:] - y[:,:-1])**2
    accion = (accion*(N / 4)).sum(axis = 1)
    
    return accion
