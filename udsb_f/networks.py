__author__ = "Matteo Pariset"

import jax
from jax.nn import sigmoid, silu, leaky_relu
import jax.numpy as jnp
import haiku as hk
import matplotlib.pyplot as plt
import optax
from tqdm.auto import tqdm
from functools import partial
from typing import NamedTuple, List, Tuple, Callable
import ipywidgets

from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics.pairwise import rbf_kernel

import pandas as pd
import joblib

import os

from scipy import linalg
import matplotlib as mpl
import itertools
import numpy

import json

from utils import is_forward


## Networks

def get_timestep_embedding(
        timesteps: jnp.ndarray,
        embedding_dim: int,
        max_positions=10000
    ) -> jnp.ndarray:
    """ Get timesteps embedding.
    Function extracted from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    Args:
        timesteps (jnp.ndarray): timesteps array (Nbatch,).
        embedding_dim (int): Size of the embedding.
        max_positions (int, optional): _description_. Defaults to 10000.

    Returns:
        emb (jnp.ndarray): embedded timesteps (Nbatch, embedding_dim).
    """
    assert embedding_dim > 3, "half_dim == 0"
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = jnp.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps[:, jnp.newaxis] * emb[jnp.newaxis, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def MLP(hidden_shapes, output_shape, bias=True, activate_final=True):
    w_init, b_init = None, None
    return hk.nets.MLP(hidden_shapes + [output_shape], with_bias=bias, w_init=w_init, b_init=b_init, activation=silu, activate_final=activate_final)

class Naive(hk.Module):
    """The standard network used to parameterize forward/backward drifts in UDSB experiments.

    Parameters:
        - `output_shape` (int): output shape.
        - `enc_shapes` (int): The shapes of the encoder.
        - `t_dim` (int): the dimension of the time embedding.
        - `dec_shapes` (int): The shapes of the decoder 
        - `resnet` (bool): if True then the network is a resnet.
    """

    def __init__(
        self,
        output_shape: int,
        enc_shapes: List[int],
        t_dim: int,
        dec_shapes: List[int],
        resnet: bool
    ):
        super().__init__()
        self.temb_dim = t_dim
        t_enc_dim = t_dim * 2

        self.output_shape = output_shape

        self.net = MLP(
            hidden_shapes=dec_shapes,
            output_shape=output_shape,
            bias=True,
            activate_final=False,
        )

        self.t_encoder = MLP(
            hidden_shapes=enc_shapes,
            output_shape=t_enc_dim,
            bias=True,
            activate_final=True,
        )

        self.x_encoder = MLP(
            hidden_shapes=enc_shapes,
            output_shape=t_enc_dim,
            bias=True,
            activate_final=True,
        )

        self.bias = hk.Bias(bias_dims=[-1])

        self.resnet = resnet

    def __call__(self, t, x):
        t = jnp.array(t, dtype=float)

        if len(x.shape) == 1:
            x_input = jnp.expand_dims(x, axis=0)
        else:
            x_input = x

        temb = get_timestep_embedding(t.reshape(-1), self.temb_dim)

        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x_input)

        temb = jnp.broadcast_to(temb, [xemb.shape[0], *temb.shape[1:]])
        h = jnp.concatenate([xemb, temb], -1)

        out = self.net(h)

        out = self.bias(out)

        if self.resnet:
            out = x_input + out

        if self.output_shape == 1:
            out = jnp.squeeze(out, axis=1)

        if len(x.shape) == 1:
            out = jnp.squeeze(out, axis=0)

        return out

def smooth_interval_indicator(x, low=0., high=1., steepness=5.):
    return sigmoid(-steepness*(x-low)) + sigmoid(steepness*(x-high))


def init_model(statespace_dim, hidden_dim_size):
    def _query_model(t, x):
        return Naive(
            output_shape=statespace_dim,
            enc_shapes=[hidden_dim_size, hidden_dim_size],
            t_dim=16,
            dec_shapes=[hidden_dim_size, hidden_dim_size],
            resnet=False
        )(t, x)
    return _query_model


class Ferryman(hk.Module):
    """The standard Ferryman network used in UDSB experiments.

    Parameters:
        - `t_dim` (int): the dimension of the time embedding.
        - `hidden_dims` (list[int]): The shape of hidden layers
        - `activate_final` (bool): if True a non-linearity is placed at the output of the network
    """
    def __init__(
        self,
        t_dim: int,
        hidden_dims: List[int],
        activate_final: bool,
    ):
        super().__init__()
        self.temb_dim = t_dim

        self.t_net = hk.nets.MLP(
            hidden_dims + [1],
            with_bias=True,
            activation=leaky_relu,
            activate_final=activate_final
        )

    def __call__(self, t, direction):
        t = jnp.array(t, dtype=float)

        # time_delay = 0 if is_forward(direction) else +1.
        time_delay = 0.

        t_emb = get_timestep_embedding(time_delay + t.reshape(-1), self.temb_dim)
        out = self.t_net(t_emb)

        return jnp.abs(out.ravel())


def init_ferryman_model(hidden_dims, activate_final=True):
    def _query_model(t, direction):
        return Ferryman(
            t_dim=32,
            hidden_dims=hidden_dims,
            activate_final=activate_final
        )(t, direction)
    return _query_model