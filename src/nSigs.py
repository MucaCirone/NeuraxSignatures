import jax
from jax import grad, jit
from jax import lax
import jax.numpy as jnp
from functools import partial
import numpy as np

###########################################################
# The vector fields 
###########################################################

@partial(jax.jit, static_argnames=['activation'])
def vec_fields(activation, AA, bb, dx, y):
    """
    Applies the vector field
        V(y) := \sum_{i=1}^d ( A_i@activation(y) + b_i )*dx^i

    :param activation: (1) -> (1)
    :param AA: (N, N, d)
    :param bb: (N, d)
    :param dx: (d)
    :param y: (N,)
    :return: V(y) : (N,) 
    """

    return (AA@dx)@activation(y) + bb@dx

@partial(jax.jit, static_argnames=['activation'])
def all_forward(activation, AA, bb, dx, Y_0):
    """
    Computes the path solving the CDE
        dY_t = \sum_{i=1}^d ( A_i@activation(Y_t) + b_i )*dx^i_t
    with initial value Y_0

    :param activation: (1) -> (1)
    :param AA: (N, N, d)
    :param bb: (N, d)
    :param dx: (times, d)
    :param Y_0: (N,)
    :return: Y : (times, N) 
    """

    times = dx.shape[-2] + 1
    Y_t = jnp.zeros(shape=(times, AA.shape[0])).at[0, :].set(Y_0)
    body_fun = lambda t, array: array.at[t+1, :].set(array[t] + vec_fields(activation, AA, bb, dx[t], array[t]))
    return lax.fori_loop(0, times-1, body_fun, Y_t)


###########################################################
# The nSig class
###########################################################

class nSig:
    def __init__(self, S_0, AA, bb, activation) -> None:
        """
        Neural Signature Class

        :param S_0: Initial value : (N,)
        :param AA: (N, N, d)
        :param bb: (N, d)
        :param activation: (1) -> (1)
        """

        # Hyperparameters
        self.N, self.d  = bb.shape
        self.S_0, self.AA, self.bb = S_0, AA, bb
        self.activation = activation

    def forward(self, X):
        """
        Computes the Neural Signature of the paths in x

        :param X: (batch, times, d)
        :return: nSig : (batch, times, N)
        """

        dX = jnp.diff(X, axis=-2)
        return jax.vmap(lambda dx_i: all_forward(self.activation, self.AA, self.bb, dx_i, self.S_0))(dX)
    
    def Gram(self, X, Y, same=False):
        """
        Computes the nSig Gram Matrix of the paths of X and Y
            Gram[i, j, s, t] := < nSig_X[i, s], nSig_Y[j, t] >_{R^N}

        :param X: (batch_x, times_x, d)
        :param Y: (batch_y, times_y, d)
        :param same: if True treat Y = X
        :return: Gram : (batch_x, batch_y, times_x, times_y)
        """
        
        # nSig_*: (batch_*, times_*, N)
        nSig_X = self.forward(X)/self.N
        
        if same:
            nSig_Y = nSig_X
        else: 
            nSig_Y = self.forward(Y)/self.N
        
        # Gram[i, j, s, t] (rSig_X[i, s, *, *, :]*rSig_Y[*, *, j, t, :]).sum(axis=-1)
        Gram = (nSig_X[:, None, :, None]*nSig_Y[None, :, None]).sum(axis=-1)
        
        return Gram
    
    def Gram_precomputed(self, nSig_X, nSig_Y):
        """
        Computes the nSig Gram Matrix of the paths of X and Y
            Gram[i, j, s, t] := < nSig_X[i, s], nSig_Y[j, t] >_{R^N}

        :param nSig_X: (batch_x, times_x, N)
        :param nSig_Y: (batch_y, times_y, N)
        :param same: if True treat Y = X
        :return: Gram : (batch_x, batch_y, times_x, times_y)
        """
        # Gram[i, j, s, t] (rSig_X[i, s, *, *, :]*rSig_Y[*, *, j, t, :]).sum(axis=-1)
        Gram = (nSig_X[:, None, :, None]*nSig_Y[None, :, None]).sum(axis=-1)
        
        return Gram






