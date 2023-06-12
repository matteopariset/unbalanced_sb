__author__ = "Matteo Pariset"

import jax
from jax import grad, vmap # type: ignore
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import haiku as hk
import optax
from functools import partial
from typing import NamedTuple, List, Tuple, Callable
from typing import List

from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics.pairwise import rbf_kernel

import pandas as pd
import joblib

import os
import sys

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
        if name == "__iter__":
            return self._cfg.__iter__
        elif name == "export" or name[0] == "_":
            return super().__getattribute__(name)
        elif name in self._cfg:
            return self._cfg[name]
        else:
            return self._root_cfg.__getattribute__(name)
        
def is_in(config, name):
    # TODO: Fix iterability of ExperimentConfig
    return name in config.__iter__()

def get_or_default(config, name, default):
    if is_in(config, name):
        return config.__getattribute__(name)
    else:
        return default

class ExperimentConfig(Config):
    
    objective = "divergence"

    times: List[int]
    mass: List[int]

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
        dict_cfg = dict(self._cfg)
        return {
            "root_cfg": dict(self._root_cfg.export()),
            "dict_cfg": dict(map(lambda k: (k, list(dict_cfg[k]) if isinstance(dict_cfg[k], omegaconf.ListConfig) else dict_cfg[k]), dict_cfg)),
        }


class Experiment:
    """ Class representing a (trained) model, together with a dataset. 
    Experiments can be saved to file and reloaded for further training or inference.
    """

    def score(self, model):
        Z = lambda params, key, t, pos: self.e.g(t, pos) * model.apply(params, key, t, pos)

        return Z
    

    def __init__(self, e: ExperimentConfig, key=None, params=None, psi=None) -> None:

        # TODO: Debug. Sample more points (now only 300) to compute Gaussian Mixture
        X_0 = e.pi_0_sample(random.PRNGKey(1))
        X_1 = e.pi_1_sample(random.PRNGKey(1))

        assert X_0.shape[-1] == e.state_dims, f"Conflicting state dimensions. Expected {e.state_dims}, got {X_0.shape}"

        bgm_0 = BayesianGaussianMixture(n_components=e.init_components_num)
        bgm_1 = BayesianGaussianMixture(n_components=e.end_components_num)

        bgm_0.fit(X_0)
        bgm_1.fit(X_1)

        # TODO: Not needed, just for debug purposes
        self.bgm_0 = bgm_0
        self.bgm_1 = bgm_1

        mass_max = max(e.mass)

        # Transform density in a JIT-compatible pdf and apply the right normalization factor
        pdf_0 = as_jax_pdf(bgm_0, total_mass=e.mass[0]/mass_max)
        pdf_1 = as_jax_pdf(bgm_1, total_mass=e.mass[-1]/mass_max)

        # TODO: Debug. Should also save pdf for full reproducibility
        pdf = {
            FORWARD:  pdf_0,
            BACKWARD: pdf_1
        }

        killing_func = killing_funcs[e.killing_function_name.replace("-", "_")]

        splitting_births_frac = get_or_default(e, "splitting_births_frac", 0.)


        sde = SDE(e, f=e.f, g=e.g, pdf=pdf, killer=killing_func, splitting_births_frac=splitting_births_frac)


        start_marginals_sampler = {
            FORWARD: partial(e.pi_0_sample, n_samples=e.batch_size),
            BACKWARD: partial(e.pi_1_sample, n_samples=e.batch_size)
        }

        hidden_dim_size = e.hidden_dims
        # TODO: Debug. The dimension of hidden layers should be > than the dimension of the state
        assert hidden_dim_size > e.state_dims, "The network hidden size should be bigger than the dimension of the state space"

        statespace_dim = X_0.shape[-1]

        model = {
            FORWARD: hk.transform(init_model(statespace_dim, hidden_dim_size)),
            BACKWARD: hk.transform(init_model(statespace_dim, hidden_dim_size))
        }

        self.e = e

        # Setup relevant entities
        self.pdf = pdf
        self.sde = sde
        self.start_marginals_sampler = start_marginals_sampler
        self.model = model

        ferryman_layers = get_or_default(self.e, "ferryman_layers_num", 3)
        activate_final = get_or_default(self.e, "ferryman_activate_final", True)

        self.ferryman = hk.transform(init_ferryman_model([64] * ferryman_layers, activate_final))

        # Create state vector
        self.experiment_state = (key, params, psi)


    def get_params(self, model_name=None):
        """ Get model parameters
        """
        _, params, _ = self.experiment_state

        if model_name is not None:
            return params[model_name] # type: ignore

        return params

    def get_psi(self):
        """ Only used by `UDSB-TD`
        """
        _, _, psi = self.experiment_state
        return psi


    def init_ipf_loss(self, sde, model, direction, objective: str):
        """ Initialize the IPF (MM) loss
        """
        if is_forward(direction):
            sign = +1
        else:
            sign = -1

        ipf_mask_dead = self.e.ipf_mask_dead

        times = jnp.array(self.e.times)
        mass = jnp.array(self.e.mass)

        def _mean_matching_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num):
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

            key, key_model = random.split(key)

            preds = (sde.f(t, pos) + sign * sde.g(t, pos) * self.score(model[direction])(params_train, key_model, t, pos)) / steps_num
            vals = pos_k_plus_1 - pos_k

            mse_loss_vec = jnp.sqrt(jnp.sum(jnp.square(preds - vals), axis=-1))

            if ipf_mask_dead:
                interval_indicator = (times[:-1] <= k) * (k < times[1:])
                mass_delta = jnp.abs((interval_indicator * mass[1:]).sum() - (interval_indicator * mass[:-1]).sum()) / mass.max()
                # alive_mask = jnp.logical_or(statuses.at[k].get(), random.uniform(key, statuses.shape[1:]) < (1. - mass_delta))
                alive_mask = jnp.logical_or(statuses.at[k].get(), statuses[:-1].at[jnp.mod(k-sign, steps_num+1)].get() * random.uniform(key, statuses.shape[1:]) < (1. - mass_delta))
            else:
                alive_mask = jnp.ones_like(statuses[k]).astype(bool)

            mse_loss = jnp.sum(alive_mask * mse_loss_vec) / jnp.clip(alive_mask.sum(), 1)

            return mse_loss

        def _divergence_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num):
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

            vec_obj = (.5 * jnp.sum(jnp.square(Z_train_value), axis=1) + divergence(key_div, _divergence_arg, t, pos) + jnp.sum(Z_eval_value * Z_train_value, axis=1)) / steps_num
            
            if ipf_mask_dead:
                alive_mask = statuses.at[k].get()
            else:
                alive_mask = jnp.ones_like(statuses[k]).astype(bool)

            alive_num = jnp.clip(alive_mask.sum(), 1)
            
            # obj = jnp.mean(vec_obj)
            obj = jnp.sum(alive_mask * vec_obj) / alive_num

            return obj

        def _combined_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num):
            return .8 * _mean_matching_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num) + .2 * _divergence_objective(params_train, params_eval, key, k, pos_k, pos_k_plus_1, statuses, steps_num)

        if objective == "mean_matching":
            return _mean_matching_objective
        elif objective == "divergence":
            return _divergence_objective
        elif objective == "combined":
            return _combined_objective
        else:
            raise ValueError(f"Unknown training objective: {objective}")


    def init_ferryman_loss(self, sde, ferryman):
        """ Initialize the Ferryman loss
        """

        assert len(self.e.times) == len(self.e.mass), "The number of mass checkpoints should equal to the number of times"
        assert len(self.e.times) >= 2, "Must at least specify mass variation over one interval"
        assert 0 <= min(self.e.times), "Times should be >= 0"
        assert max(self.e.times) <= self.e.steps_num, "Times should be >= 0"
        assert ((jnp.array(self.e.times)[1:] - jnp.array(self.e.times)[:-1]) > 0).all()

        steps_num = self.e.steps_num
        intervals_num = len(self.e.times)-1

        assignment_matrix = jnp.zeros((intervals_num, steps_num+1))

        for int_num, (b, e) in enumerate(zip(self.e.times[:-1], self.e.times[1:])):
            assignment_matrix = assignment_matrix.at[int_num,b:e].set(1.)

        mass_max = max(self.e.mass)

        norm_masses = jnp.array(self.e.mass) / mass_max

        # TODO: Ddebug. Birth by splitting are not accounted for when reality_coeff is big (i.e. alive_to_alive should contribute to births)
        reality_coeff = get_or_default(self.e, "reality_coefficient", .1)

        def _ferryman_loss(params_train, key, trajectory_direction, trajs, statuses, td_coeff):
            """ Ferryman Loss (Ours)

            Used to learn the network Q
            """

            time = jnp.linspace(0, 1, steps_num+1)

            if is_forward(trajectory_direction):
                death_sign = 1
                m_init, m_end = norm_masses[:-1], norm_masses[1:]
                interval_assignment = assignment_matrix
            else:
                death_sign = -1
                # Flip the trajectory 
                trajs = trajs[::-1]
                statuses = statuses[::-1]
                time = time[::-1]
                m_init, m_end = norm_masses[::-1][:-1], norm_masses[::-1][1:]
                interval_assignment = assignment_matrix[::-1,::-1]

            statuses = jnp.concatenate([statuses[:1], statuses[:-1]], axis=0)


            interval_assignment = interval_assignment[:,:-1]

            key, key_seed = random.split(key)
            key_model = random.split(key_seed, steps_num+1)

            eval_ferryman = partial(ferryman.apply, params=params_train, direction=trajectory_direction)

            prev_trajs = jnp.concatenate([trajs[:1], trajs[:-1]], axis=0)


            death_threshold = vmap(lambda t, x: sde.killer(t, x))(time, trajs)
            infinite_barriers = jnp.isinf(death_threshold)[:-1]

            raw_death_rates = vmap(
                lambda key, t, prev_x, prev_status, x, death_rate: death_sign * sde.reweight_killing(key, trajectory_direction, death_rate, eval_ferryman, prev_x, prev_status, t, x, clip=False)[1]
            )(key_model, time, prev_trajs, statuses, trajs, jnp.nan_to_num(death_threshold, posinf=0.))[:-1]

            death_rates = jnp.logical_not(infinite_barriers) * jnp.clip(raw_death_rates, -1., 1.) + infinite_barriers * 1.


            alive = statuses[:-1]
            alive_to_dead = jnp.logical_and(alive, jnp.logical_not(statuses[1:]))
            all_deaths = alive * jnp.clip(death_rates, 0.)
            real_deaths = alive_to_dead * jnp.clip(death_rates, 0.)

            dead = jnp.logical_not(statuses[:-1])
            dead_to_alive = jnp.logical_and(dead, statuses[1:])

            # Account for birth by splitting (i.e., births from living particles)
            key, key_death_type = random.split(key)
            splitting_birth_selector = random.uniform(key_death_type, shape=dead.shape) < sde.splitting_births_frac
            can_give_birth = jnp.logical_or(dead * jnp.logical_not(splitting_birth_selector), alive * splitting_birth_selector)

            all_births = can_give_birth * jnp.clip(death_rates, None, 0.)
            real_births = dead_to_alive * jnp.clip(death_rates, None, 0.)

            transitions_num = jnp.clip((reality_coeff * (alive_to_dead + dead_to_alive).astype(float) + (1-reality_coeff) * (alive + dead).astype(float)).sum(axis=1), 1.)

            # # Compute change of mass contraints
            # change_of_mass_constraint = jnp.abs(((possible_deaths + possible_births).sum(axis=1) / transitions_num).sum(axis=0) - (m_end[0]-m_init[-1]))

            transitions = reality_coeff * (real_deaths + real_births) + (1-reality_coeff) * (all_deaths + all_births)
            predicted_mass_variations = jnp.matmul(interval_assignment, (transitions.sum(axis=1) / transitions_num))

            change_of_mass_constraint = jnp.abs(m_init[0] - jnp.cumsum(predicted_mass_variations) - m_end).sum()

            # Killing rate regularization
            out_of_bounds_reg = (jnp.clip(jnp.abs(smooth_interval_indicator(raw_death_rates, -1.0, 1.0, 30) * raw_death_rates), 1.) - 1.).sum() / jnp.clip(jnp.logical_not(infinite_barriers).sum(), 1.)

            return (td_coeff > 0.) * (change_of_mass_constraint) + out_of_bounds_reg

        return _ferryman_loss


    def _check_status(self):
        return sum(map(lambda c: c is None, self.experiment_state)) == 0

    @staticmethod
    def create(config, key=None, params=None, psi=None):
        """ Create a fresh experiment.
            
            Parameters:
              - `config`: experiment configuration, see `toy_experiments.ipynb` for examples
              - `key`: random number generator. If `None`, the RNG is initialized with seed 0
              - `params`: model parameters. If `None`, parameters are initialized to random values
              - `psi`: only used by `UDSB-TD`
        """
        dataset_name = config["dataset"]["dataset_name"]
        experiment_config = ExperimentConfig(datasets[dataset_name](**config["dataset"]), omegaconf.OmegaConf.create(config["experiment"]))

        return Experiment(experiment_config, key, params, psi)

    @staticmethod
    def load(dataset_name, tag):
        """ Reload saved experiment.
        
            Parameters:
              - `dataset_name`: identifier of the dataset
              - `tag`: experiment name tag
        """
        with open(get_config_file(dataset_name, tag), 'r') as f:
            full_config = json.load(f)
            key, params, psi = load_experiment(dataset_name, tag)

            return Experiment.create(full_config, key, params, psi)


    def save(self, tag=""):
        """ Save experiment (configuration and model weights) to file.
        
            Parameters:
              - `tag`: experiment name tag
        """
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
