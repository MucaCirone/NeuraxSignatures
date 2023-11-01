import numpy as np
import jax
from jax import grad, jit
from jax import lax
from jax import random
import jax.numpy as jnp

from jax.experimental import sparse
from scipy.special import gamma

from functools import partial

from jax import random
key = random.PRNGKey(0)

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


###########################################################
# Random Matrix Generators for plug and play
###########################################################

def Gaussian_weights(key, N, d, bias=False):
  """
  Produce Gaussian Matrices

  :param key: jax.randm key
  :param N: hidden dimension
  :param d: driver dimension
  :param bias: if False then bb=0
  :return: key, S_0, AA, bb
  """

  key, *subkeys = random.split(key, 4)

  S_0 = jax.random.normal(subkeys[0], shape=(N,))
  AA = 1/np.sqrt(N)*jax.random.normal(subkeys[1], shape=(N, N, d))

  if bias:
    bb = jax.random.normal(subkeys[2], shape=(N, d))
  else:
    bb = jnp.zeros(shape=(N, d))
    
  return (key, S_0, AA, bb)

def Weibull_weights(key, N, d, scale=1, concentration=4):
  """
  Produce Gaussian Matrices

  :param key: jax.randm key
  :param N: hidden dimension
  :param d: driver dimension
  :param scale: Weibull hyperparam
  :param concentration: Weibull hyperparam
  :return: key, S_0, AA, bb
  """

  key, *subkeys = random.split(key, 4)

  S_0 = jax.random.normal(subkeys[0], shape=(N,))

  mean = scale*gamma(1+1/concentration)
  std = scale*jnp.sqrt(gamma(1+2/concentration) - gamma(1+1/concentration)**2)
  AA = - 1/np.sqrt(N)*(jax.random.weibull_min(subkeys[1], scale, concentration, shape=(N, N, d)) - mean)/std

  bb = jnp.zeros(shape=(N, d))

  return (key, S_0, AA, bb)

def LogNormal_weights(key, N, d, sigma=.3):
  """
  Produce LogNormal Matrices

  :param key: jax.randm key
  :param N: hidden dimension
  :param d: driver dimension
  :param sigma: LogNormal hyperparam
  :return: key, S_0, AA, bb
  """ 

  key, *subkeys = random.split(key, 4)

  S_0 = jax.random.normal(subkeys[0], shape=(N,))

  mean = jnp.exp(0.5*(sigma**2))
  std = jnp.exp(0.5*(sigma**2))*jnp.sqrt(jnp.exp((sigma**2))-1)
  AA = 1/np.sqrt(N)*(jax.random.lognormal(subkeys[1], sigma, shape=(N, N, d)) - mean)/std

  bb = jnp.zeros(shape=(N, d))

  return (key, S_0, AA, bb)

def sparsify_weights(key, S_0, AA, bb, verbose=False):
  """
  Make the Matrices sparse

  :param key: jax.randm key
  :param N: hidden dimension
  :param d: driver dimension
  :param verbose: if True print Sparsity factor 
  :return: key, S_0, AA, bb
  """ 
  key, *subkeys = random.split(key, 2)

  N, d = bb.shape
  pN_inverse = np.sqrt(N*np.sqrt(N*np.sqrt(N)))

  mask = jax.random.bernoulli(subkeys[0], p=1/pN_inverse, shape=(N, N, d))
  AA = AA*mask*np.sqrt(pN_inverse)

  # If sparse enough use sparse matrix format
  if pN_inverse > 200:
    AA = sparse.BCOO.fromdense(AA)

  # If verbose print sparsity coeff
  if verbose:
    print("Sparsity: ", 1/pN_inverse)

  return (key, S_0, AA, bb)


###########################################################
# Ploting Helpers
###########################################################

def plot_Gram(G):
  """
  Plot the Gram dependent on 2 time params

  :param G: (times_x, times_y)
  """ 
  
  times_x, times_y = G.shape
  
  fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": "3d"})

  # Make data.
  X = np.arange(0, times_x, 1)
  Y = np.arange(0, times_y, 1)
  X, Y = np.meshgrid(X, Y)
  Z = G
  
  # Plot the surface.
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

  # Customize the z axis.
  ax.zaxis.set_major_locator(LinearLocator(10))
  # A StrMethodFormatter is used automatically
  ax.zaxis.set_major_formatter('{x:.02f}')
  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)