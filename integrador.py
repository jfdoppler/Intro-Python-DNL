#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:04:29 2019

@author: juan
"""

import numpy as np
from scipy.integrate import odeint, ode
import matplotlib.pyplot as plt


# %% Sistema de ecuaciones
def f(t, z):
    x = z[0]
    y = z[1]
    dxdt = x-y
    dydt = x**2-4
    return [dxdt, dydt]


def J(x, y):
    return np.matrix(([1, -1], [2*x, 0]))


# %% Nulclinas
XX, YY = np.meshgrid(np.arange(-10, 10, .01), np.arange(-10, 10, .01))
DX, DY = f(0, [XX, YY])

nulx = plt.contour(XX, YY, DX, levels=[0], colors='red', linestyles='dashed')
plt.clabel(nulx, nulx.levels, fmt='$\dot{x}=0$')
nuly = plt.contour(XX, YY, DY, levels=[0], colors='black', linestyles='dashed')
plt.clabel(nuly, nuly.levels, fmt='$\dot{y}=0$')

# %% Odeint
dt = 0.001
tmax = 10
t = np.arange(0, tmax, dt)
tpre = -10
tant = np.arange(0, tpre, -dt)
Xi = np.linspace(-4, 4, 4)
Yi = np.linspace(-4, 4, 4)
#plt.figure()
for xi in Xi:
    for yi in Yi:
        plt.scatter(xi, yi)
        zi = [xi, yi]
        sol_fut = odeint(f, zi, t, tfirst=True)
        sol_pas = odeint(f, zi, tant, tfirst=True)
        x_fut = sol_fut[:, 0]
        x_pas = sol_pas[:, 0][::-1]
        xt = np.concatenate((x_pas, x_fut))
        y_fut = sol_fut[:, 1]
        y_pas = sol_pas[:, 1][::-1]
        yt = np.concatenate((y_pas, y_fut))
        plt.plot(xt, yt)
X = np.linspace(-10, 10, 8)
Y = np.linspace(-10, 10, 8)
XX, YY = np.meshgrid(X, Y)
DX, DY = f(0, [XX, YY])
plt.xlim(-7, 7)
plt.ylim(-7, 7)
# %% Streamplot
plt.streamplot(XX, YY, DX, DY, density=.5, minlength=.1)

# %% con rk
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
                  length_includes_head=True, head_width=.4, color=c)
        solver.set_initial_value(zi, 0)
        xant = np.zeros_like(tant)
        xant.fill(np.nan)
        yant = np.zeros_like(tant)
        yant.fill(np.nan)
        for ix, tt in enumerate(tant):
            xant[ix], yant[ix] = solver.integrate(tant[ix])
        plt.plot(xant, yant, color=c)
plt.xlim(-7, 7)
plt.ylim(-7, 7)


# %% Linealizacion
puntos_fijos = [[-2, -2], [2, 2]]
ci_var = []
dist = 0.01
for pf in puntos_fijos:
    xpf = pf[0]
    ypf = pf[1]
    eig_val, eig_vec = np.linalg.eig(J(xpf, ypf))
    if not isinstance(eig_val[0], complex):
        for ix, eig in enumerate(eig_val):
            vec = np.transpose(np.asarray(eig_vec[:, ix]))[0]
            ci_var.append([xpf-dist*eig*vec[0], ypf-dist*eig*vec[1]])
            ci_var.append([xpf+dist*eig*vec[0], ypf+dist*eig*vec[1]])
            plt.plot([xpf-eig*vec[0], xpf+eig*vec[0]], [ypf-eig*vec[1], 
                      ypf+eig*vec[1]],'--r')
            plt.arrow(xpf+np.abs(eig)*vec[0]/2, ypf+np.abs(eig)*vec[1]/2,
                      eig*vec[0]/4, eig*vec[1]/4,
                      shape='full', lw=0, length_includes_head=True,
                      head_width=.4, color='r')
            plt.arrow(xpf-np.abs(eig)*vec[0]/2, ypf-np.abs(eig)*vec[1]/2,
                      -eig*vec[0]/4, -eig*vec[1]/4,
                      shape='full', lw=0, length_includes_head=True,
                      head_width=.4, color='r')
plt.plot([x[0] for x in ci_var], [x[1] for x in ci_var], '.')

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