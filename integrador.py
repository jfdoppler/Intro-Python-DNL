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


# %% Nulclinas
XX, YY = np.meshgrid(np.arange(-10, 10, .01), np.arange(-10, 10, .01))
DX, DY = f(0, [XX, YY])

nulx = plt.contour(XX, YY, DX, levels=[0], colors='red', linestyles='dashed')
plt.clabel(nulx, nulx.levels, fmt='$\dot{x}=0$')
nuly = plt.contour(XX, YY, DY, levels=[0], colors='black', linestyles='dashed')
plt.clabel(nuly, nuly.levels, fmt='$\dot{y}=0$')

# %% Puntos fijos
def abs_deriv(x, y):
    dx, dy = f(0, [x, y])
    return(np.sqrt(dx**2+dy**2))

im = plt.imshow(DX**2+DY**2, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(-10, 10, -10, 10))
# %%
dt = 0.001
tmax = 10
t = np.arange(0, tmax, dt)
tpre = -10
tant = np.arange(0, tpre, -dt)
Xi = np.linspace(-4, 4, 4)
Yi = np.linspace(-4, 4, 4)
plt.figure()
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
plt.streamplot(XX, YY, DX, DY, density=.5, minlength=.1)
plt.xlim(-10, 10)
plt.ylim(-10, 10)
# %% con rk
dt = 0.01
tmax = 10
t = np.arange(0, tmax, dt)
tpre = -10
solver = ode(f).set_integrator('dopri5')
tant = np.arange(0, tpre, -dt)
Xi = np.linspace(-4, 4, 4)
Yi = np.linspace(-4, 4, 4)
plt.figure()
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
plt.xlim(-10, 10)
plt.ylim(-10, 10)


# %% Linealizacion
def J(x, y):
    return np.matrix(([1, -1], [2*x, 0]))


fp = [[-2, -2], [2, 2]]