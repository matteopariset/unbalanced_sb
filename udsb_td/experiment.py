__author__ = "Matteo Pariset"

import jax
import jax.random as random
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
import sys

from scipy import linalg
import matplotlib as mpl
import itertools
import numpy

import json
import omegaconf
import abc
import datetime

from utils import *
from datasets import *
from networks import *
from sde import *


class Config():

    def __init__(self, root_cfg, cfg) -> None:
        self._root_cfg = root_cfg 
        self._cfg = cfg

    def __getattribute__(self, name: str):
        if name == "export" or name[0] == "_":
            return super().__getattribute__(name)
        elif name in self._cfg:
            return self._cfg[name]
        else:
            return self._root_cfg.__getattribute__(name)

class ExperimentConfig(Config):
    
    objective = "divergence"

    mass_0: int
    mass_1: int

    init_components_num: int
    end_components_num: int

    steps_num: int
    batch_size: int
    paths_reuse: int
    hidden_dims: int

    eps: int
    neginf: int
    plusinf: int

    experiment_name: str

    def export(self):
        return {
            "root_cfg": dict(self._root_cfg.export()),
            "dict_cfg": dict(self._cfg),
        }


class Experiment:

    @staticmethod
    def density(model):
        return model.apply

    def score(self, model, broadcast_batch=True):
        log_density = Experiment.density(model)
        grad_log_density = jax.grad(log_density, argnums=3)
        if broadcast_batch:
            grad_log_density = jax.vmap(grad_log_density, in_axes=(None, None, None, 0), out_axes=(0))

        Z = lambda params, key, t, pos: self.e.g(t, pos) * grad_log_density(params, key, t, pos)

        return Z


    def __init__(self, e: ExperimentConfig, key=None, params=None, psi=None) -> None:

        # TODO: Debug. Sample more points (now only 300) to compute Gaussian Mixture
        X_0 = e.pi_0_sample(random.PRNGKey(1))
        X_1 = e.pi_1_sample(random.PRNGKey(1))

        # timeframe = {
        #     2: X_0,
        #     4: X_1
        # }

        # initial_points = X_0
        # final_points = X_1

        assert X_0.shape[-1] == e.state_dims, f"Conflicting state dimensions. Expected {e.state_dims}, got {X_0.shape}"
        assert e.batch_size <= e.mass_0, "Batch size exceeds total mass"

        bgm_0 = BayesianGaussianMixture(n_components=e.init_components_num)
        bgm_1 = BayesianGaussianMixture(n_components=e.end_components_num)

        bgm_0.fit(X_0)
        bgm_1.fit(X_1)

        def get_normalized_mass(direction):
            if is_forward(direction):
                return min(1., e.mass_0/e.mass_1)
            else:
                return min(1., e.mass_1/e.mass_0)

        # TODO: Not needed, just for debug purposes
        self.bgm_0 = bgm_0
        self.bgm_1 = bgm_1

        # Transform density in a JIT-compatible pdf and apply the right normalization factor
        pdf_0 = as_jax_pdf(bgm_0, total_mass=get_normalized_mass(FORWARD))
        pdf_1 = as_jax_pdf(bgm_1, total_mass=get_normalized_mass(BACKWARD))

        # TODO: Debug. Should also save pdf for full reproducibility
        pdf = {
            FORWARD:  pdf_0,
            BACKWARD: pdf_1
        }

        killing_func = killing_funcs[e.killing_function_name.replace("-", "_")]


        sde = SDE(e, f=e.f, g=e.g, pdf=pdf, killer=killing_func)


        start_marginals_sampler = {
            FORWARD: partial(e.pi_0_sample, n_samples=e.batch_size),
            BACKWARD: partial(e.pi_1_sample, n_samples=e.batch_size)
        }

        hidden_dim_size = e.hidden_dims
        # TODO: Debug. The dimension of hidden layers should be > than the dimension of the state
        assert hidden_dim_size > e.state_dims, "The network hidden size should be bigger than the dimension of the state space"

        model = {
            FORWARD: hk.transform(init_model(hidden_dim_size)),
            BACKWARD: hk.transform(init_model(hidden_dim_size))
        }

        self.e = e

        # Setup relevant entities
        self.pdf = pdf
        self.sde = sde
        self.start_marginals_sampler = start_marginals_sampler
        self.model = model

        # Create state vector
        self.experiment_state = (key, params, psi)


    def get_params(self):
        _, params, _ = self.experiment_state
        return params

    def get_psi(self):
        _, _, psi = self.experiment_state
        return psi


    def init_ipf_loss(self, sde, model, direction, objective: str):
        if is_forward(direction):
            sign = +1
        else:
            sign = -1


        def _mean_matching_objective(params_train, params_eval, key, psi, k, pos_k, pos_k_plus_1, statuses, steps_num):
            """ Mean-matching objective _(De Bortoli 2021)_

            `loss(Z) = || (X_{k+1} - X_k) - (f ± gZ) Δt ||`
            """

            t = k/steps_num

            # Because of how integrals are discretized, `pos` corresponds to:
            #  - X[k]: for the FORWARD direction
            #  - X[k+1]: for the BACKWARD direction
            if is_forward(direction):
                pos = pos_k
            else:
                pos = pos_k_plus_1

            preds = (sde.f(t, pos) + sign * sde.g(t, pos) * self.score(model[direction])(params_train, key, t, pos)) / steps_num
            vals = pos_k_plus_1 - pos_k

            mse_loss_vec = jnp.sqrt(jnp.sum(jnp.square(preds - vals), axis=-1))

            # alive_mask = statuses.at[k].get()
            alive_mask = jnp.ones_like(statuses[k]).astype(bool)
            
            mse_loss = jnp.sum(alive_mask * mse_loss_vec) / jnp.clip(alive_mask.sum(), 1)

            return mse_loss

        def _divergence_objective(params_train, params_eval, key, psi, k, pos_k, pos_k_plus_1, statuses, steps_num):
            """ Divergence-based objective. Inspired by _(TChen 2021)_
                but including missing terms to make its magnitude comparable to other losses in _(Liu 2022)_
            
            FORWARD:
            `loss(Z) = 1/2 ||Z||^2 + ∇·(f + gZ) + <Z_hat, Z> - V...`

            BACKWARD:
            `loss(Z_hat) = 1/2 ||Z_hat||^2 + ∇·(-f + gZ_hat) + <Z, Z_hat> + V...`
            """

            t = k/steps_num

            Z_train, Z_eval = self.score(model[direction]), self.score(model[reverse(direction)])

            if is_forward(direction):
                pos = pos_k
                params_forward = params_train
            else:
                pos = pos_k_plus_1
                params_forward = params_eval

            key, key_z_train, key_div, key_z_eval = random.split(key, 4)

            Z_train_value = Z_train(params_train, key_z_train, t, pos)
            Z_eval_value = Z_eval(params_eval, key_z_eval, t, pos)

            def _divergence_arg(t, pos):
                return sign * sde.f(t, pos) + sde.g(t, pos) * Z_train(params_train, key_z_train, t, pos)

            # vec_obj = (.5 * jnp.sum(jnp.square(Z_train_value), axis=1) + divergence(key_div, _divergence_arg, t, pos) + jnp.sum(Z_eval_value * Z_train_value, axis=1)) / steps_num - sign * jnp.clip(sde.killer(t, pos) * psi.at[0].get()/jnp.exp(sde.validate_y(self.density(model[FORWARD])(params_forward, key_z_train, t, pos))), -1., 1.)
            vec_obj = (.5 * jnp.sum(jnp.square(Z_train_value), axis=1) + divergence(key_div, _divergence_arg, t, pos) + jnp.sum(Z_eval_value * Z_train_value, axis=1)) / steps_num - jnp.clip(sde.killer(t, pos) * psi.at[0].get()/jnp.exp(sde.validate_y(self.density(model[FORWARD])(params_forward, key_z_train, t, pos))), -1., 1.)
            
            # alive_mask = statuses.at[k].get()
            alive_mask = jnp.ones_like(statuses[k]).astype(bool)

            alive_num = jnp.clip(alive_mask.sum(), 1)
            
            # obj = jnp.mean(vec_obj)
            obj = jnp.sum(alive_mask * vec_obj) / alive_num

            return obj

        def _combined_objective(params_train, params_eval, key, psi, k, pos_k, pos_k_plus_1, statuses, steps_num):
            return .3 * _mean_matching_objective(params_train, params_eval, key, psi, k, pos_k, pos_k_plus_1, statuses, steps_num) + .7 * _divergence_objective(params_train, params_eval, key, psi, k, pos_k, pos_k_plus_1, statuses, steps_num)

        if objective == "mean_matching":
            return _mean_matching_objective
        elif objective == "divergence":
            return _divergence_objective
        elif objective == "combined":
            return _combined_objective
        else:
            raise ValueError(f"Unknown training objective: {objective}")


    def init_td_loss(self, sde, model, direction):

        def _td_loss(params_train, key, psi, k, traj, ys, statuses):
            """ Temporal Difference loss _(Liu 2022)_ with regularization (only for forward model)

            FORWARD:
            `loss(Y) = || Y_{k} - y_{k} ||`

            FORWARD:
            `loss(Y_hat) = || Y_hat_{k} - y_hat_{k} ||`
            """

            key, key_model, key_random_timestep = random.split(key, 3)

            pos = traj.at[k].get()

            preds = self.density(model[direction])(params_train, key_model, k/self.e.steps_num, pos)
            vals = self.sde.validate_y(ys.at[k].get())

            alive_mask = statuses.at[k].get()

            alive_num = jnp.clip(alive_mask.sum(), 1)

            # TODO: Debug. Should only use alive trajectories?
            td_loss = jnp.sum(alive_mask * jnp.abs(preds - vals)) / alive_num
            # td_loss = jnp.mean(jnp.abs(preds - vals))

            loss = td_loss

            return loss

        return _td_loss


    def _check_status(self):
        return sum(map(lambda c: c is None, self.experiment_state)) == 0

    @staticmethod
    def create(config, key=None, params=None, psi=None):
        dataset_name = config["dataset"]["dataset_name"]
        experiment_config = ExperimentConfig(datasets[dataset_name](**config["dataset"]), omegaconf.OmegaConf.create(config["experiment"]))

        return Experiment(experiment_config, key, params, psi)

    @staticmethod
    def load(dataset_name, tag):
        with open(get_config_file(dataset_name, tag), 'r') as f:
            full_config = json.load(f)
            key, params, psi = load_experiment(dataset_name, tag)

            return Experiment.create(full_config, key, params, psi)


    def save(self, tag=""):
        if tag == "":
            tag = self.e.experiment_name + "__" + datetime.datetime.today().strftime('%Y_%m_%d-%H_%M_%S')

        with open(get_config_file(self.e.dataset_name, tag), "w") as f:
            exported_cfg = self.e.export()
            json.dump({
                "dataset": exported_cfg['root_cfg'],
                "experiment": exported_cfg['dict_cfg'],
            }, f)

        if self._check_status():
            key, params, psi = self.experiment_state
            save_experiment(self.e.dataset_name, tag, key, params, psi)
        else:
            # raise ValueError("Invalid experiment state. Load it from file or by performing training")
            info("Saving only configuration. Experiment state not available")
