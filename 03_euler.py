
# coding: utf-8

# # Integración I
# 
# ## Método de Euler
# 

# Supongamos que tenemos una ecuación diferencial de esta forma:
# $$
# \frac{df}{dx} = g(x)
# $$
# 
# El método de Euler consiste en aproximar la derivada por el cociente incremental $\frac{df}{dx} \simeq \frac{\Delta f}{\Delta x}$:
# 
# $$
# \frac{f(x+\Delta x) - f(x)}{\Delta x} = g(x)
# $$
# 
# $$
# f(x+\Delta x) = f(x) + g(x) \Delta x
# $$
# 
# Nos queda definida la x siguiente en función de la condición inicial.
# 
# #### Ejemplo
# Integremos el modelo de Gompertz (ej. 3 de la guía 1)
# 
# $$
# \dot{N} = -aNln(bN)
# $$
# 
# $$
# \frac{N(t + \Delta t) - N(t)}{\Delta t} = -aN(t) ln(bN(t))
# $$
# 
# $$
# N(t + \Delta t) = N(t) -aN(t) ln(bN(t))\Delta t
# $$
# 
# Noten que para resolver esto tenemos que definir, además de la condición incial, cual es el paso temporal. Qué les parece que puede pasar si cambiamos este valor?

# In[10]:

import numpy as np
import matplotlib.pyplot as plt


# In[9]:

a = 1
b = 1
dt = 0.1
t = np.arange(0, 10, dt)
N = np.zeros_like(t)

N[0] = 3
for i in range(len(t)-1):
    N[i + 1] = -a*np.log(b*N[i]) * dt + N[i]

plt.plot(t, N)


# ## 2D
# 
# Consideren el siguiente sistema:
# 
# $$
# \dot{x} = 4x+2y
# $$
# $$
# \dot{y} = -17x-5y
# $$
# 
# Qué tipo de punto fijo es el origen?

# In[18]:

dt = 0.01

t = np.arange(0, 10, dt)
x = np.zeros_like(t)
y = np.zeros_like(t)

x[0] = 1.5
y[0] = 0.
for i in range(len(t)-1):
    x[i + 1] = x[i] + (4*x[i] + 2*y[i]) * dt
    y[i + 1] = y[i] + (-17*x[i]-5*y[i]) * dt

plt.plot(x, y)


# Integre el sistema del ejercicio 6e de la guía 3:
# 
# $$
# \dot{x} = 5x+2y
# $$
# $$
# \dot{y} = -17x-5y
# $$
# 
# Qué tipo de punto fijo es el origen? Cómo da la integración numérica?
