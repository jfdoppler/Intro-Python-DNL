#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:04:29 2019

@author: juan
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt


# %% Sistema de ecuaciones
def f(t, z):
    x = z[0]
    y = z[1]
    dxdt = x+np.exp(-y)
    dydt = -y
    return [dxdt, dydt]


def J(x, y):
    return np.matrix(([1, -np.exp(-y)], [0, -1]))


# %% Nulclinas
XX, YY = np.meshgrid(np.arange(-10, 10, .01), np.arange(-10, 10, .01))
DX, DY = f(0, [XX, YY])

plt.figure(figsize=(10, 6))
nulx = plt.contour(XX, YY, DX, levels=[0], colors='magenta', linestyles='dotted')
plt.clabel(nulx, nulx.levels, fmt='$\dot{x}=0$')
nuly = plt.contour(XX, YY, DY, levels=[0], colors='black', linestyles='dotted')
plt.clabel(nuly, nuly.levels, fmt='$\dot{y}=0$')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
# %% Linealizacion
puntos_fijos = [[-1, 0]]
ci_var = []
dist = 0.01
for pf in puntos_fijos:
    xpf = pf[0]
    ypf = pf[1]
    plt.plot(xpf, ypf, 'o', c='b')
    eig_val, eig_vec = np.linalg.eig(J(xpf, ypf))
    if not isinstance(eig_val[0], complex):
        for ix, eig in enumerate(eig_val):
            vec = np.transpose(np.asarray(eig_vec[:, ix]))[0]
            print('Autovalor = {}; autovec = {}'.format(eig, vec))
            ci_var.append([xpf-dist*eig*vec[0], ypf-dist*eig*vec[1]])
            ci_var.append([xpf+dist*eig*vec[0], ypf+dist*eig*vec[1]])
            plt.plot([xpf-eig*vec[0], xpf+eig*vec[0]], [ypf-eig*vec[1], 
                      ypf+eig*vec[1]],'--r')
            plt.plot([xpf-100*vec[0], xpf+100*vec[0]], [ypf-100*vec[1], 
                      ypf+100*vec[1]],'r', ls='dotted')
            plt.arrow(xpf+np.abs(eig)*vec[0]/2, ypf+np.abs(eig)*vec[1]/2,
                      eig*vec[0]/4, eig*vec[1]/4,
                      shape='full', lw=0, length_includes_head=True,
                      head_width=.3, color='r')
            plt.arrow(xpf-np.abs(eig)*vec[0]/2, ypf-np.abs(eig)*vec[1]/2,
                      -eig*vec[0]/4, -eig*vec[1]/4,
                      shape='full', lw=0, length_includes_head=True,
                      head_width=.3, color='r')

# %% con rk y miramos nodo-silla
dt = 0.01
tmax = 10
t = np.arange(0, tmax, dt)
tpre = -10
solver = ode(f).set_integrator('dopri5')
tant = np.arange(0, tpre, -dt)
Xi = np.linspace(-4, 4, 4)
Yi = np.linspace(-4, 4, 4)
for xi in Xi:
    for yi in Yi:
        zi = [xi, yi]
        solver.set_initial_value(zi, 0)
        xt = np.zeros_like(t)
        xt.fill(np.nan)
        yt = np.zeros_like(t)
        yt.fill(np.nan)
        for ix, tt in enumerate(t):
            xt[ix], yt[ix] = solver.integrate(t[ix])
        line = plt.plot(xt, yt)
        c = line[0].get_color()
        dxi, dyi = f(0, zi)
        plt.arrow(xi, yi, dxi*dt, dyi*dt, shape='full', lw=0,
                  length_includes_head=True, head_width=.3, color=c)
        solver.set_initial_value(zi, 0)
        xant = np.zeros_like(tant)
        xant.fill(np.nan)
        yant = np.zeros_like(tant)
        yant.fill(np.nan)
        for ix, tt in enumerate(tant):
            xant[ix], yant[ix] = solver.integrate(tant[ix])
        plt.plot(xant, yant, color=c)

# %% Todo
plt.xlim(-7, 7)
plt.ylim(-7, 7)

# %% Streamplot
plt.streamplot(XX, YY, DX, DY, density=1, minlength=.1)

# %% Variedad (ci cerca)
for zi in ci_var:
    solver.set_initial_value(zi, 0)
    xt = np.zeros_like(t)
    xt.fill(np.nan)
    yt = np.zeros_like(t)
    yt.fill(np.nan)
    for ix, tt in enumerate(t):
        xt[ix], yt[ix] = solver.integrate(t[ix])
    plt.plot(xt, yt, color='k', lw=2)
    solver.set_initial_value(zi, 0)
    xant = np.zeros_like(tant)
    xant.fill(np.nan)
    yant = np.zeros_like(tant)
    yant.fill(np.nan)
    for ix, tt in enumerate(tant):
        xant[ix], yant[ix] = solver.integrate(tant[ix])
    plt.plot(xant, yant, color='k', lw=2)
# %%
xx = np.linspace(-10, 10, 1000)
coefs = [0, 2, 4/3, -38/27]
for lastix in range(len(coefs)):
    ix = 0
    yy = np.zeros_like(xx)
    while ix <= lastix:
        yy += coefs[ix]*(xx+1)**(ix)
        ix += 1
    plt.plot(xx, yy, ls='dashed', label='Orden {}'.format(lastix))
plt.xlim(-1.5, -0.5)
plt.ylim(-0.5, 0.5)
plt.legend()