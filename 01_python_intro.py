
# coding: utf-8

# ## Intro a python/programación
# 
# Python es un lenguaje de programación libre y gratuito. Como es libre y gratuito mucha gente, empresas, organizaciones, hacen paquetes (conjuntos de rutinas) de python.
# 
# ### Para bajar
# 
# La forma más fácil de tener Python es bajar [Anaconda](https://www.anaconda.com/distribution/), que viene con varios paquetes ya instalados.
# 
# ## IDEs
# 
# Una [IDE](https://en.wikipedia.org/wiki/Integrated_development_environment) es un entorno de desarrollo integrado. Anaconda viene con Jupyter Notebook (web, .ipynb) y Spyder (para los que usaron Matlab, la más familiar).
# 
# ### Jupyter Notebook
# 
# Se divide en celdas (que tienen un "In [ ]" al costado) donde se escribe el código. Con Ctrl + Enter o el botoncito que parece un play arriba, se ejecuta el código de la celda.
# 
# Con el "+" de arriba o Esc+b, se agrega una celda.
# 
# ### Google y Stackoverflow son sus amigos
# 
# La comunidad de programadores (y de Python) es enorme. Por eso, la primera fuente de información es internet. La regla básica es que cualquier problema con el que se encuentren, ya se le habrá presentado a un montón de gente, y por lo general googlear lo que se quiere hacer o el mensaje de error cuando algo falla, es una herramienta fundamental.
# 
# ### Hola mundo!
# 

# In[25]:

print('Hola mundo!')


# ### Aritmética básica
# Se pueden hacer las operaciones básicas.

# In[5]:

1+1


# In[6]:

8*5


# Para controlar que se imprime, se puede usar la función print.
# 
# #### Qué hacen los operadores //, % y **?

# In[8]:

print(7/2)
print(7//2)
print(7%2)
print(5**2)
print(25**0.5)


# Podemos guardar resultados en variables. No hace falta (como en C) que le digamos qué tipo de variable es (un entero, un float, string...), y podemos cambiar de tipo.

# In[24]:

a = 3
b = 2
lala = a*b
print(lala)
a = str(b)
a


# Atención! A la derecha del igual se hacen todas las cuentas y el resultado se guarda en el símbolo de la izquierda

# In[12]:

c = 2
d = 5
c = c*d
print(c)


# ## Listas, tuplas y diccionarios
# 
# Hasta ahora vimos tres tipos de datos, los números enteros (int, de integer en inglés), los racionales (float, de números de punto flotante) y los strings (str, texto)
# 
# Las listas se definen usando corchetes, con sus elementos (que pueden ser cualquier cosa, incluso otras listas) separados por comas.
# 
# Las listas tienen orden, y se puede acceder a sus elementos por el índice (la numeración empieza por 0)

# In[33]:

lista1 = [1, 4, 0]
lista2 = [4, 'hola', [4,5], lista1, lista1[2], 10]
print(lista1)
print(lista2[1])
lista1[1] = -5
print(lista1)


# También podemos acceder "desde el final", usando indices negativos.

# In[35]:

print(lista1[-1])
print(lista2[-3])


# Y quedarnos con un pedazo (desde:hasta, incluyendo el desde, excluyendo el hasta)

# In[36]:

lista2[1:3]


# Las tuplas son parecidas a las listas, salvo que son inmutables (no se pueden cambiar sus valores). En vez de corchetes, van con paréntesis. Ojo, para crear una tupla de 1 elemento, hay que poner una coma después

# In[42]:

tupla = (1, 2, 'a')
print(tupla)
print(tupla[1])
print(tupla[-1])
tupla2 = (2, )
print(type(tupla2))
not_tupla = (2)
print(type(not_tupla))


# In[40]:

tupla[1] = 2


# Los diccionarios son como las listas pero cada entrada viene de a dos: key y value (como palabra y definición en el diccionario). Se definen con llaves

# In[44]:

dic = {'a': 1, 'b': [2, 4]}


# Se accede a los valores usando las keys, y se le pueden agregar entradas.

# In[45]:

print(dic['b'])
dic['c'] = 'hola'
print(dic)


# ## Loops
# 
# Un loop es una manera de escribir mucho código en muy poco espacio, cuando queremos que una operación se repita muchas veces

# In[47]:

for i in lista1:
    print(i)


# # If, elif, else
# Con esto podemos definir pedazos de código que solo se ejecutan si se cumple una condición

# In[48]:

x = 0
if x < 0:
    print('x < 0')
elif x > 0:
    print('x > 0')
else:
    print('x = 0')   


# ## While
# Nos permite definir código que se ejecuta (una y otra vez...) mientras una condición se cumpla

# In[49]:

n = 0
while n < 5:
    n += 1
print(n)


# # Funciones, args, kwargs
# Cuando queremos ejecutar un código muchas veces, suele convenir definir funciones. Las funciones toman un valor de entrada, hacen alguna operación y nos devuelven otro valor.

# In[52]:

def flip(string):
    return string[::-1]


# In[53]:

flip('hola')


# In[58]:

def lineal(x, ordenada=1, pendiente=1):
    return pendiente*x+ordenada

print(lineal(5))
print(lineal(5, ordenada=4))
print(lineal(5, ordenada=4, pendiente=-1))


# ## Paquetes
# Una de las fortalezas de Python es que hay mucha gente muy comprometida en hacer código compartible. El código se comparte en forma de paquetes (o librerías). Algunos famosos son numpy (numérico), scipy (científico?), matplotlib (gráficos), pandas (manejo de datos)
# 
# Para poder usar los paquetes hay que importarlos. Hay distintas formas de importar:

# In[54]:

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import *


# In[56]:

x = np.arange(0, 10, 0.1)
y = x**2
s = signal.triang(len(x))*100
print(x.mean())
print(y.std())
plt.plot(x, y)
plt.plot(x, s)
plt.show()

