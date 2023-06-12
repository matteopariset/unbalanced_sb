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

from utils import *
from networks import *
from experiment import *


## Experiment functions (SDE)

class SDE:
    """
    Models forward & backward SDEs:
    (notation from T. Chen & al., 2022)

    FORWARD:
      `dX_t = (f(t, X_t) + g²(t, X_t) ∇log 𝜑(t, X_t)) dt + g(t, X_t) dB_t`
    BACKWARD:
      `dY_t = (f(t, Y_t) - g²(t, Y_t) ∇log 𝜑_{hat}(t, Y_t)) dt + g(t, Y_t) dB_t`
      where the minus sign in the second formula corresponds to the backward direction.

    Note that:
      - `∀t: (X_t)# . == (Y_t)# .`
      - `density` and `score` are NNs that parametrize log 𝜑 and g∇log 𝜑, respectively
    """

    def __init__(self, config, f, g, pdf, killer: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = None) -> None:
        self.config = config
        self.f = f
        self.g = g
        self.pdf = pdf

        # TODO: Support stateful killing functions
        self.killer = killer

        # TODO: Make parameter
        # Available choices: "keep", "freeze"
        self.dead_mode = "keep"

    def get_normalized_mass(self, direction):
        if is_forward(direction):
            return min(1., self.config.mass_0/self.config.mass_1)
        else:
            return min(1., self.config.mass_1/self.config.mass_0)

    def validate_y(self, ys):
        """ Ensure that log 𝜑 and log 𝜑_{hat} do not become too small (or too big)
        """
        # TODO: Fix a meaningful upper-limit to log Ψ: e.g. max{log(pdf_i(x_i))}
        return jnp.clip(jnp.nan_to_num(ys, neginf=self.config.neginf), self.config.neginf, self.config.plusinf)

    def langevin_correct(self, key, model, t, pos):
        """ Applies Langevin correction, as detailed in (TChen 2021)
        """
        r = 0.005
        def _langevin_step(i, l_args):
            key, corrected_pos = l_args

            key, key_score, key_score_hat, key_epsilon = random.split(key, 4)
            
            epsilon_i = random.normal(key_epsilon, shape=corrected_pos.shape)

            Z_t_i = model[FORWARD](key_score, t, corrected_pos)
            Z_hat_t_i = model[BACKWARD](key_score_hat, t, corrected_pos)

            sigma_t_i = jnp.expand_dims(2 * jnp.square(r * self.g(t, corrected_pos)) * jnp.sum(jnp.square(epsilon_i), axis=1) / (jnp.sum(jnp.square(Z_t_i + Z_hat_t_i), axis=1)), axis=1)  # shape: (n,1)

            score_t_i = (Z_t_i + Z_hat_t_i) / self.g(t, corrected_pos)
            corrected_pos = corrected_pos + sigma_t_i * score_t_i + jnp.sqrt(2 * sigma_t_i) * epsilon_i

            return key, corrected_pos
        
        key, next_pos = jax.lax.fori_loop(0, 20, _langevin_step, (key, pos))

        return key, next_pos

    def compute_initial_y(self, key, direction, density, t_init, x_init):
        
        # Exploit invariance by rescaling of (𝜑, 𝜑_{hat}) pair, i.e. log 𝜑 ≈ (log ρ)/2
        log_rho = jnp.log(self.pdf[direction](x_init))
        y_init_from_reverse = log_rho - self.validate_y(density[reverse(direction)](key, t_init, x_init))
        
        # # TODO: Debug. Rethink additive bias also when starting from t=1
        # if not is_forward(direction):
        #     y_init_from_reverse = (y_init_from_reverse - y_init_from_reverse.mean()) + log_rho.mean()
        # # y_init_from_reverse = (y_init_from_reverse - y_init_from_reverse.mean()) - 10

        # if is_forward(direction):
        #     y_init_from_reverse = (y_init_from_reverse - y_init_from_reverse.mean()) + log_rho.mean() - 10
        # else:
        #     y_init_from_reverse = (y_init_from_reverse - y_init_from_reverse.mean()) + log_rho.mean() + 10

        return self.validate_y(y_init_from_reverse)
        

    def compute_y_increment(self, key, direction, density, score, rand, psi, t, pos):
        """ Compute the infinitesimal change of log-densities log 𝜑  (resp. log 𝜑_{hat}), corresponding to the FORWARD (resp. BACKWARD) directions.

            The increment is defined as:

            FORWARD:
              `dY(t, X_t) = (1/2 ||Z(t, Xh_t)||^2  +  ∇·[g(t) Z(t, Xh_t)]  +  <Z(t, Xh_t), Z_{hat}(t, Xh_t)> - V + V ψ/exp(Y(t, Xh_t))) dt  +  <Z(t, Xh_t), dB_t>`
            BACKWARD:
              `dY_{hat}(t, X_t) = (1/2 ||Z_{hat}(t, X_t)||^2  +  ∇·[-f(t, X_t) + g(t) Z_{hat}(t, X_t)]  +  <Z_{hat}(t, X_t), Z(t, X_t)> - V) dt  +  <Z_{hat}(t, X_t), dB_t>`

            where `Y = log 𝜑` and `Z = g ∇log 𝜑`.

            Note that:
              - `rand` is not a standard key, but a dictionary containing the keys already used to compute the same quantities used for trajectory calculation
        """

        # TODO: Debug. Update docs

        dt = 1/self.config.steps_num

        dB_t = jnp.sqrt(dt) * random.normal(rand['dB'], pos.shape)

        curr_Z = score[FORWARD](rand['Z'], t, pos)
        curr_Z_hat = score[BACKWARD](rand['Z_hat'], t, pos)

        key, key_divergence = random.split(key)

        psi_val = psi.at[0].get()

        if is_forward(direction):
            def _divergence_arg(t, pos):
                return self.f(t, pos) + self.g(t, pos) * score[FORWARD](rand['Z'], t, pos)

            # dY_t = (.5 * jnp.sum(jnp.square(curr_Z), axis=1) + divergence(key_divergence, _divergence_arg, t, pos) + jnp.sum(curr_Z * curr_Z_hat, axis=1)) * dt - jnp.clip(self.killer(t, pos), -1., 1.) + jnp.clip(self.killer(t, pos) * psi_val/jnp.exp(self.validate_y(density[FORWARD](rand['Z'], t, pos))), -1., 1.) + jnp.sum(curr_Z * dB_t, axis=1)
            dY_t = .5 * jnp.sum(jnp.square(curr_Z), axis=1) * dt + jnp.clip(self.killer(t, pos), -1., 1.) - jnp.clip(self.killer(t, pos) * psi_val/jnp.exp(self.validate_y(density[FORWARD](rand['Z'], t, pos))), -1., 1.) + jnp.sum(curr_Z * dB_t, axis=1)
        else:
            def _divergence_arg(t, pos):
                return self.f(t, pos) + self.g(t, pos) * score[BACKWARD](rand['Z_hat'], t, pos)
                
            # dY_t = (.5 * jnp.sum(jnp.square(curr_Z_hat), axis=1) + divergence(key_divergence, _divergence_arg, t, pos) + jnp.sum(curr_Z_hat * curr_Z, axis=1)) * dt - jnp.clip(self.killer(t, pos), -1., 1.) + jnp.sum(curr_Z_hat * dB_t, axis=1)
            dY_t = .5 * jnp.sum(jnp.square(curr_Z_hat), axis=1) * dt + jnp.clip(self.killer(t, pos), -1., 1.) + jnp.sum(curr_Z_hat * dB_t, axis=1)
            # dY_t = -dY_t
            
        return dY_t

    def sample_y_given_trajectory(self, key, direction, density, score, psi, traj, statuses):
        if is_forward(direction):
            k_init = 0
            t_init = 0.
        else:
            k_init = -1
            t_init = 1.
        
        x_init = traj.at[k_init].get()

        key, key_y_init = random.split(key)

        ys_increments = jnp.zeros(traj.shape[:-1])
        ys_increments = ys_increments.at[k_init].set(self.compute_initial_y(key_y_init, direction, density, t_init, x_init))

        def _f(k, args):
            if not is_forward(direction):
                k = self.config.steps_num - k

            key, part_ys_incs = args

            key, key_func, *key_rand = random.split(key, 5)
            
            # TODO: Improve this
            rand = {
                "Z": key_rand[0],
                "Z_hat": key_rand[1],
                "dB": key_rand[2]
            }

            t = k/self.config.steps_num

            part_ys_incs = part_ys_incs.at[k].set(self.compute_y_increment(key_func, direction, density, score, rand, psi, t, traj.at[k].get()))

            return key, part_ys_incs

        key, ys_increments = jax.lax.fori_loop(1, self.config.steps_num+1, _f, (key, ys_increments))

        if is_forward(direction):
            ys = jnp.cumsum(ys_increments, axis=0)
        else:
            # Must sum BACKWARD in time
            ys = jnp.cumsum(ys_increments.at[::-1].get(), axis=0).at[::-1].get()

        return self.validate_y(ys)

    def apply_killer(self, key, death_threshold):
        key, key_deaths, key_births = random.split(key, 3)
        deaths = random.uniform(key_deaths, (death_threshold.shape[0],)) < death_threshold
        births = -random.uniform(key_births, (death_threshold.shape[0],)) < -1-death_threshold

        return key, deaths, births

    def reweight_killing(self, key, direction, density, death_threshold, psi, t, pos, status):
        # TODO: Write docs

        key, key_killing_reweighting = random.split(key)

        # TODO: Debug. Symmetrize this for birth-only forward
        if is_forward(direction):
            death_reweighting = psi.at[0].get() / jnp.exp(self.validate_y(density[FORWARD](key_killing_reweighting, t, pos)))
        else:
            K_t = (1-status.mean())
            death_reweighting = psi.at[0].get() / (jnp.exp(self.validate_y(density[FORWARD](key_killing_reweighting, t, pos))) * K_t)
        
        death_threshold = jnp.clip(death_threshold * death_reweighting, -1., 1.)

        return key, death_threshold

    def sample_f_trajectory(self, key, x_0, density, score, psi, corrector):
        steps_num = self.config.steps_num

        traj = jnp.zeros((steps_num+1, *x_0.shape))
        traj = traj.at[0].set(x_0)

        key, key_init_status = random.split(key)

        statuses = jnp.zeros(traj.shape[:-1]).astype(bool)
        # TODO: Debug. Discuss if case mass_0 < mass_1 is sound in our framework
        # Start with right proportion of dead/alive particles
        statuses = statuses.at[0].set(random.uniform(key_init_status, shape=(x_0.shape[0],)) < self.get_normalized_mass(FORWARD))

        key, key_density_b_init = random.split(key)

        ys = jnp.zeros((steps_num+1, x_0.shape[0]))
        ys = ys.at[0].set(self.compute_initial_y(key_density_b_init, FORWARD, density, 0., x_0))

        def _step(i, args):
            key, curr_pos, part_traj, part_statuses, part_y = args

            key, key_score, key_score_hat, key_brownian, key_y = random.split(key, 5)
            rand = {
                'Z': key_score,
                'Z_hat': key_score_hat,
                'dB': key_brownian
            }

            t = (i-1)/steps_num
            dt = 1/steps_num
            curr_g = self.g(t, curr_pos)
            curr_score = score[FORWARD](key_score, t, curr_pos)

            next_pos = curr_pos + (self.f(t, curr_pos) + curr_g * curr_score) * dt + curr_g * jnp.sqrt(dt) * random.normal(key_brownian, x_0.shape)

            next_y = part_y.at[i-1].get() + self.compute_y_increment(key_y, FORWARD, density, score, rand, psi, t, curr_pos)
            part_y = part_y.at[i].set(next_y)

            if corrector == "langevin":
                key, next_pos = self.langevin_correct(key, score, t, next_pos)

            if self.killer is not None:
                death_threshold = self.killer(t, next_pos)
                key, death_threshold = self.reweight_killing(key, FORWARD, density, death_threshold, psi, t, next_pos, part_statuses.at[i-1].get())
                
                key, dead_mask, birth_mask = self.apply_killer(key, death_threshold)

                curr_status = part_statuses.at[i-1].get()

                # DEATHS
                curr_status = jnp.logical_and(curr_status, jnp.logical_not(dead_mask))

                # BIRTHS
                # ...from shadow paths
                curr_status = jnp.logical_or(curr_status, jnp.logical_and(jnp.logical_not(curr_status), birth_mask))

                # # ...from splitting
                # key, curr_status, next_pos = birth_by_splitting(key, curr_status, birth_mask, next_pos)

                part_statuses = part_statuses.at[i].set(curr_status)

                if self.dead_mode == "keep":
                    pass
                elif self.dead_mode == "freeze":
                    raise NotImplementedError("Cannot freeze trajectories")

            part_traj = part_traj.at[i].set(next_pos)

            return key, next_pos, part_traj, part_statuses, part_y

        key, x_1, traj, statuses, ys = jax.lax.fori_loop(1, steps_num+1, _step, (key, x_0, traj, statuses, ys))

        ys = self.validate_y(ys)

        return traj, ys, statuses


    manifold_constraint = True

    def compute_y_increment_evo(self, key, direction, density, score, rand, psi, t, pos):
        """ Compute the infinitesimal change of log-densities log 𝜑  (resp. log 𝜑_{hat}), corresponding to the FORWARD (resp. BACKWARD) directions.

            The increment is defined as:

            FORWARD:
              `dY(t, X_t) = (1/2 ||Z(t, Xh_t)||^2  +  ∇·[g(t) Z(t, Xh_t)]  +  <Z(t, Xh_t), Z_{hat}(t, Xh_t)> - V + V ψ/exp(Y(t, Xh_t))) dt  +  <Z(t, Xh_t), dB_t>`
            BACKWARD:
              `dY_{hat}(t, X_t) = (1/2 ||Z_{hat}(t, X_t)||^2  +  ∇·[-f(t, X_t) + g(t) Z_{hat}(t, X_t)]  +  <Z_{hat}(t, X_t), Z(t, X_t)> - V) dt  +  <Z_{hat}(t, X_t), dB_t>`

            where `Y = log 𝜑` and `Z = g ∇log 𝜑`.

            Note that:
              - `rand` is not a standard key, but a dictionary containing the keys already used to compute the same quantities used for trajectory calculation
        """

        dt = 1/self.config.steps_num

        dB_t = jnp.sqrt(dt) * random.normal(rand['dB'], pos.shape)

        curr_Z = score[FORWARD](rand['Z'], t, pos)
        curr_Z_hat = score[BACKWARD](rand['Z_hat'], t, pos)

        key, key_divergence = random.split(key)

        psi_val = psi.at[0].get()

        # if SDE.manifold_constraint:
        #     penalty_discount = 2.
        #     variants_frac = pos[:,2:]
        #     death_rate = jnp.clip(jnp.square(variants_frac.sum(axis=1) - 4)/penalty_discount, 0., 1.)
        #     V_ext = self.killer(t, pos) + death_rate
        # else:
        #     V_ext = self.killer(t, pos)
        V_ext = self.killer(t, pos)
        
        if is_forward(direction):
            def _divergence_arg(t, pos):
                return self.f(t, pos) + self.g(t, pos) * score[FORWARD](rand['Z'], t, pos)

            dY_t = (.5 * jnp.sum(jnp.square(curr_Z), axis=1) + divergence(key_divergence, _divergence_arg, t, pos) + jnp.sum(curr_Z * curr_Z_hat, axis=1)) * dt - jnp.clip(V_ext, -1., 1.) + jnp.clip(self.killer(t, pos) * psi_val/jnp.exp(self.validate_y(density[FORWARD](rand['Z'], t, pos))), -1., 1.) + jnp.sum(curr_Z * dB_t, axis=1)
        else:
            def _divergence_arg(t, pos):
                return -self.f(t, pos) + self.g(t, pos) * score[BACKWARD](rand['Z_hat'], t, pos)
                
            dY_t = (.5 * jnp.sum(jnp.square(curr_Z_hat), axis=1) + divergence(key_divergence, _divergence_arg, t, pos) + jnp.sum(curr_Z_hat * curr_Z, axis=1)) * dt - jnp.clip(V_ext, -1., 1.) + jnp.sum(curr_Z_hat * dB_t, axis=1)
            
        return dY_t

    def compute_initial_y_evo(self, key, direction, density_type, density, x_init):
        if is_forward(direction):
            t_init = 0.
        else:
            t_init = 1.
        log_rho = jnp.log(self.pdf[direction](x_init))
        y_init_from_reverse = log_rho - self.validate_y(density[reverse(density_type)](key, t_init, x_init))

        # if not is_forward(density_type):
        #     # 𝜑_hat(0) should be centered around the marginal density
        #     y_init_from_reverse = (y_init_from_reverse - y_init_from_reverse.mean()) + log_rho.mean()
    
        return self.validate_y(y_init_from_reverse)

    def sample_f_trajectory_evo(self, key, x_0, density, score, psi, corrector):
        steps_num = self.config.steps_num

        traj = jnp.zeros((steps_num+1, *x_0.shape))
        traj = traj.at[0].set(x_0)

        key, key_init_status = random.split(key)

        statuses = jnp.zeros(traj.shape[:-1]).astype(bool)
        # TODO: Debug. Discuss if case mass_0 < mass_1 is sound in our framework
        # Start with right proportion of dead/alive particles
        statuses = statuses.at[0].set(random.uniform(key_init_status, shape=(x_0.shape[0],)) < self.get_normalized_mass(FORWARD))

        key, key_density_b_init = random.split(key)

        ys_hat = jnp.zeros((steps_num+1, x_0.shape[0]))
        ys_hat = ys_hat.at[0].set(self.compute_initial_y_evo(key_density_b_init, FORWARD, BACKWARD, density, x_0))

        def _step(i, args):
            key, curr_pos, part_traj, part_statuses, part_y_hat = args

            key, key_score, key_score_hat, key_brownian, key_y = random.split(key, 5)
            rand = {
                'Z': key_score,
                'Z_hat': key_score_hat,
                'dB': key_brownian
            }

            t = (i-1)/steps_num
            dt = 1/steps_num
            curr_g = self.g(t, curr_pos)
            curr_score = score[FORWARD](key_score, t, curr_pos)

            next_pos = curr_pos + (self.f(t, curr_pos) + curr_g * curr_score) * dt + curr_g * jnp.sqrt(dt) * random.normal(key_brownian, x_0.shape)

            next_y = part_y_hat.at[i-1].get() + self.compute_y_increment_evo(key_y, BACKWARD, density, score, rand, psi, t, curr_pos)
            part_y_hat = part_y_hat.at[i].set(next_y)

            if corrector == "langevin":
                key, next_pos = self.langevin_correct(key, score, t, next_pos)

            if self.killer is not None:
                death_threshold = self.killer(t, next_pos)
                key, death_threshold = self.reweight_killing(key, FORWARD, density, death_threshold, psi, t, next_pos, part_statuses.at[i-1].get())
                
                key, dead_mask, birth_mask = self.apply_killer(key, death_threshold)

                curr_status = part_statuses.at[i-1].get()

                # DEATHS
                curr_status = jnp.logical_and(curr_status, jnp.logical_not(dead_mask))

                # BIRTHS
                # ...from shadow paths
                curr_status = jnp.logical_or(curr_status, jnp.logical_and(jnp.logical_not(curr_status), birth_mask))

                # # ...from splitting
                # key, curr_status, next_pos = birth_by_splitting(key, curr_status, birth_mask, next_pos)

                part_statuses = part_statuses.at[i].set(curr_status)

                if self.dead_mode == "keep":
                    pass
                elif self.dead_mode == "freeze":
                    raise NotImplementedError("Cannot freeze trajectories")

            part_traj = part_traj.at[i].set(next_pos)

            return key, next_pos, part_traj, part_statuses, part_y_hat

        key, x_1, traj, statuses, ys_hat = jax.lax.fori_loop(1, steps_num+1, _step, (key, x_0, traj, statuses, ys_hat))

        ys_hat = self.validate_y(ys_hat)

        return traj, ys_hat, statuses

    def sample_b_trajectory(self, key, x_1, density, score, psi, corrector):
        steps_num = self.config.steps_num

        traj = jnp.zeros((steps_num+1, *x_1.shape))
        traj = traj.at[-1].set(x_1)

        key, key_init_status = random.split(key)

        statuses = jnp.zeros(traj.shape[:-1]).astype(bool)  # shape (k+1,n)
        # Start with right proportion of dead/alive particles
        statuses = statuses.at[-1].set(random.uniform(key_init_status, shape=(x_1.shape[0],)) < self.get_normalized_mass(BACKWARD))

        birth_places = jnp.copy(x_1)  # shape (n,d)

        key, key_density_f_init = random.split(key)

        ys_hat = jnp.zeros((steps_num+1, x_1.shape[0]))
        ys_hat = ys_hat.at[-1].set(self.compute_initial_y(key_density_f_init, BACKWARD, density, 1., x_1))

        def _step(i, args):
            key, curr_pos, part_traj, part_statuses, part_birth_places, part_y_hat = args
            key, key_score, key_score_hat, key_brownian, key_y = random.split(key, 5)
            rand = {
                "Z": key_score,
                "Z_hat": key_score_hat,
                "dB": key_brownian
            }

            i = steps_num - i
            
            t = (i+1)/steps_num
            dt = 1/steps_num
            curr_g = self.g(t, curr_pos)
            curr_score = score[BACKWARD](key_score_hat, t, curr_pos)

            next_pos = curr_pos - ((self.f(t, curr_pos) - curr_g * curr_score) * dt + curr_g * jnp.sqrt(dt) * random.normal(key_brownian, x_1.shape))

            next_y = part_y_hat.at[i+1].get() + self.compute_y_increment(key_y, BACKWARD, density, score, rand, psi, t, curr_pos)
            part_y_hat = part_y_hat.at[i].set(next_y)

            if corrector == "langevin":
                key, next_pos = self.langevin_correct(key, score, t, next_pos)

            if self.killer is not None:
                death_threshold = self.killer(t, next_pos)
                key, death_threshold = self.reweight_killing(key, BACKWARD, density, death_threshold, psi, t, next_pos, part_statuses.at[i+1].get())

                key, birth_mask, dead_mask = self.apply_killer(key, death_threshold)

                curr_status = part_statuses.at[i+1].get()

                # DEATHS
                curr_status = jnp.logical_and(curr_status, jnp.logical_not(dead_mask))
                
                # BIRTHS
                # ...from shadow paths
                curr_status = jnp.logical_or(curr_status, jnp.logical_and(jnp.logical_not(curr_status), birth_mask))

                # TODO: Debug. Restore births by splitting
                # TODO: Debug. Before restoring, must take into account that ys of splitted particles SHOULD BE COPIED from the source (otherwise ys computation would not hold!!!)
                # # ...from splitting
                # key, curr_status, next_pos = birth_by_splitting(key, curr_status, birth_mask, next_pos)

                part_statuses = part_statuses.at[i].set(curr_status)

                born_now = jnp.logical_and(jnp.logical_not(part_statuses.at[i+1].get()), part_statuses.at[i].get())

                born_now_mask = jnp.expand_dims(born_now, axis=1)

                part_birth_places = born_now_mask * next_pos + jnp.logical_not(born_now_mask) * part_birth_places

            part_traj = part_traj.at[i].set(next_pos)

            return key, next_pos, part_traj, part_statuses, part_birth_places, part_y_hat

        key, x_0, traj, statuses, _, ys_hat = jax.lax.fori_loop(1, steps_num+1, _step, (key, x_1, traj, statuses, birth_places, ys_hat))

        # # TODO: Debug. Must mirror this behavior for forward trajectories (on birth processes)
        # # Convert shadow particles that were not used midway into standard trajectories
        # unused_shadow_mask = jnp.expand_dims(statuses.sum(axis=0) == 0, axis=0)  # shape (1, n)
        # statuses = jnp.logical_or(statuses, unused_shadow_mask * jnp.ones_like(statuses).astype(bool))

        ys_hat = self.validate_y(ys_hat)

        if self.dead_mode == "keep":
            pass
        elif self.dead_mode == "freeze":
            assert False, "Unimplemented"

        return traj, ys_hat, statuses

    def sample_b_trajectory_evo(self, key, x_1, density, score, psi, corrector):
        steps_num = self.config.steps_num

        traj = jnp.zeros((steps_num+1, *x_1.shape))
        traj = traj.at[-1].set(x_1)

        key, key_init_status = random.split(key)

        statuses = jnp.zeros(traj.shape[:-1]).astype(bool)  # shape (k+1,n)
        # Start with right proportion of dead/alive particles
        statuses = statuses.at[-1].set(random.uniform(key_init_status, shape=(x_1.shape[0],)) < self.get_normalized_mass(BACKWARD))

        birth_places = jnp.copy(x_1)  # shape (n,d)

        key, key_density_f_init = random.split(key)

        ys = jnp.zeros((steps_num+1, x_1.shape[0]))
        ys = ys.at[-1].set(self.compute_initial_y_evo(key_density_f_init, BACKWARD, FORWARD, density, x_1))

        def _step(i, args):
            key, curr_pos, part_traj, part_statuses, part_birth_places, part_y = args
            key, key_score, key_score_hat, key_brownian, key_y = random.split(key, 5)
            rand = {
                "Z": key_score,
                "Z_hat": key_score_hat,
                "dB": key_brownian
            }

            i = steps_num - i
            
            t = (i+1)/steps_num
            dt = 1/steps_num
            curr_g = self.g(t, curr_pos)
            curr_score = score[BACKWARD](key_score_hat, t, curr_pos)

            next_pos = curr_pos - ((self.f(t, curr_pos) - curr_g * curr_score) * dt + curr_g * jnp.sqrt(dt) * random.normal(key_brownian, x_1.shape))

            next_y = part_y.at[i+1].get() - self.compute_y_increment_evo(key_y, FORWARD, density, score, rand, psi, t, curr_pos)
            part_y = part_y.at[i].set(next_y)

            if corrector == "langevin":
                key, next_pos = self.langevin_correct(key, score, t, next_pos)

            if self.killer is not None:
                death_threshold = self.killer(t, next_pos)
                key, death_threshold = self.reweight_killing(key, BACKWARD, density, death_threshold, psi, t, next_pos, part_statuses.at[i+1].get())

                key, birth_mask, dead_mask = self.apply_killer(key, death_threshold)

                curr_status = part_statuses.at[i+1].get()

                # DEATHS
                curr_status = jnp.logical_and(curr_status, jnp.logical_not(dead_mask))
                
                # BIRTHS
                # ...from shadow paths
                curr_status = jnp.logical_or(curr_status, jnp.logical_and(jnp.logical_not(curr_status), birth_mask))

                # TODO: Debug. Restore births by splitting
                # TODO: Debug. Before restoring, must take into account that ys of splitted particles SHOULD BE COPIED from the source (otherwise ys computation would not hold!!!)
                # # ...from splitting
                # key, curr_status, next_pos = birth_by_splitting(key, curr_status, birth_mask, next_pos)

                part_statuses = part_statuses.at[i].set(curr_status)

                born_now = jnp.logical_and(jnp.logical_not(part_statuses.at[i+1].get()), part_statuses.at[i].get())

                born_now_mask = jnp.expand_dims(born_now, axis=1)

                part_birth_places = born_now_mask * next_pos + jnp.logical_not(born_now_mask) * part_birth_places

            part_traj = part_traj.at[i].set(next_pos)

            return key, next_pos, part_traj, part_statuses, part_birth_places, part_y

        key, x_0, traj, statuses, _, ys = jax.lax.fori_loop(1, steps_num+1, _step, (key, x_1, traj, statuses, birth_places, ys))

        # # TODO: Debug. Must mirror this behavior for forward trajectories (on birth processes)
        # # Convert shadow particles that were not used midway into standard trajectories
        # unused_shadow_mask = jnp.expand_dims(statuses.sum(axis=0) == 0, axis=0)  # shape (1, n)
        # statuses = jnp.logical_or(statuses, unused_shadow_mask * jnp.ones_like(statuses).astype(bool))

        ys = self.validate_y(ys)

        if self.dead_mode == "keep":
            pass
        elif self.dead_mode == "freeze":
            assert False, "Unimplemented"

        return traj, ys, statuses

    def sample_trajectory(self, key, direction, x_init, density, score, psi, corrector=""):
        if is_forward(direction):
            return self.sample_f_trajectory(key, x_init, density, score, psi, corrector)
        else:
            return self.sample_b_trajectory(key, x_init, density, score, psi, corrector)

    def sample_trajectory_evo(self, key, direction, x_init, density, score, psi, corrector=""):
        if is_forward(direction):
            return self.sample_f_trajectory_evo(key, x_init, density, score, psi, corrector)
        else:
            return self.sample_b_trajectory_evo(key, x_init, density, score, psi, corrector)
    
    def guess_psi_from_space(self, key, marginals, sampling_density, sampling_score, psi):
        """ Approximate the value of ψ as:

         `ψ = (1 - mass_1/mass_0) / sum_k( sum_n( V(k, Xn_k) exp(Y_{hat}(k, Xn_k)) ))`
        
        """
        key[FORWARD], key_f_init_points, key_f_traj = random.split(key[FORWARD], 3)
        key[BACKWARD], key_b_init_points, key_b_traj = random.split(key[BACKWARD], 3)

        f_trajs, f_ys, f_statuses = self.sample_trajectory(key_f_traj, FORWARD, marginals[FORWARD](key=key_f_init_points), sampling_density, sampling_score, psi)
        b_trajs, b_ys_hat, b_statuses = self.sample_trajectory(key_b_traj, BACKWARD, marginals[BACKWARD](key=key_b_init_points), sampling_density, sampling_score, psi)

        def _estimate_psi_denominator_from_nn(key, t, pos, status):
            # Estimate log 𝜑/𝜑_{hat}
            y_hat_estimate = sampling_density[BACKWARD](key, t, pos)

            return status * self.killer(t, pos) * jnp.exp(self.validate_y(y_hat_estimate))

        def _estimate_psi_denominator_from_sde(t, pos, y_hat, status):
            # Estimate log 𝜑_{hat}
            y_hat_estimate = y_hat

            return status * self.killer(t, pos) * jnp.exp(self.validate_y(y_hat_estimate))

        nn_estimate_psi_denominator = jax.vmap(_estimate_psi_denominator_from_nn, in_axes=(0, 0, 0, 0), out_axes=(0))
        sde_estimate_psi_denominator = jax.vmap(_estimate_psi_denominator_from_sde, in_axes=(0, 0, 0, 0), out_axes=(0))

        key[BACKWARD], *key_stock_y_hat = random.split(key[BACKWARD], self.config.steps_num+2)

        # # Inactive trajectories are defined as those which are never alive. They are de facto shrinking the batch size
        # f_active_trajs_num = (f_statuses.sum(axis=0) > 0.)
        # b_active_trajs_num = (b_statuses.sum(axis=0) > 0.)

        # # Reweight the empirical sum by the effective batch size
        # f_summation_weight = batch_size/jnp.clip(f_active_trajs_num, 1)
        # b_summation_weight = batch_size/jnp.clip(b_active_trajs_num, 1)
        f_summation_weight = 1
        b_summation_weight = 1

        # TODO: Debug. Only using forward trajectories for now
        # Approximate integral as the mean of backward and forward estimates
        time_steps = jnp.arange(0, self.config.steps_num+1, 1) / self.config.steps_num
        # integrand_val = .5 * (f_summation_weight * f_psi_denominator(jnp.array(key_stock_y_hat), time_steps, f_trajs, f_statuses) + b_summation_weight * b_psi_denominator(time_steps, b_trajs, b_ys_hat, b_statuses))
        # integrand_val = .5 * (f_summation_weight * nn_estimate_psi_denominator(jnp.array(key_stock_y_hat), time_steps, f_trajs, blur_statuses(f_statuses)) + b_summation_weight * nn_estimate_psi_denominator(jnp.array(key_stock_y_hat), time_steps, b_trajs, blur_statuses(b_statuses)))
        integrand_val = nn_estimate_psi_denominator(jnp.array(key_stock_y_hat), time_steps, f_trajs, blur_statuses(f_statuses))

        # TODO: Debug. The formula changes, if mass_0 != 1 or mass_0 < mass_1
        guessed_psi = (1-self.config.mass_1/self.config.mass_0) / jnp.clip(jnp.mean(jnp.sum(integrand_val, axis=1), axis=0), self.config.eps)

        # TODO: Impove this by turning psi into a scalar
        return key, jnp.array([guessed_psi])

    def guess_psi_from_trajs(self, key, marginals, sampling_density, sampling_score, psi):
        """ Approximate the value of ψ:

        `ψ = (1 - mass_1/mass_0) / ∫_t( ∫_x( V(t, x) 𝜑_{hat}(t, x) )dx )dt`

        Using that:

        `ρ = 𝜑_{hat}𝜑`

        is the distribution over paths of the Schroedinger Bridge, the denominator in the above expression
        can be rewritten as:

        `∫_t( ∫_x( V 𝜑_{hat} )dx )dt = ∫_t( ∫_x( V/𝜑 𝜑_{hat}𝜑 )dx )dt = ∫_t( ∫_x( V/𝜑 ρ )dx )dt`

        which corresponds to:

        `𝔼[V/𝜑]`

        where the expectation runs over paths.
        This quantity can be faithfully approximated by:

        `D = sum_k( sum_n( V(k, X^n_k) / exp(Y(k, X^n_k)) )) / (K N)`

        giving:

        `ψ ≈ (1 - mass_1/mass_0) / D`

        """
        key[FORWARD], key_f_init_points, key_f_traj = random.split(key[FORWARD], 3)
        key[BACKWARD], key_b_init_points, key_b_traj = random.split(key[BACKWARD], 3)

        f_trajs, f_ys, f_statuses = self.sample_trajectory(key_f_traj, FORWARD, marginals[FORWARD](key=key_f_init_points), sampling_density, sampling_score, psi)
        b_trajs, b_ys_hat, b_statuses = self.sample_trajectory(key_b_traj, BACKWARD, marginals[BACKWARD](key=key_b_init_points), sampling_density, sampling_score, psi)

        def _estimate_reweighted_killing(key, t, pos, status):
            # Estimate ψ/𝜑
            log_psi_nn = sampling_density[FORWARD](key, t, pos)

            return status * (self.killer(t, pos) / jnp.exp(self.validate_y(log_psi_nn))) * (1/status.sum())

        nn_estimate_reweighted_killing = jax.vmap(_estimate_reweighted_killing, in_axes=(0, 0, 0, 0), out_axes=(0))

        key[BACKWARD], *key_stock_y_hat = random.split(key[BACKWARD], self.config.steps_num+2)

        # TODO: Debug. Only using forward trajectories here
        # Approximate integral as the mean of backward and forward estimates
        time_steps = jnp.arange(0, self.config.steps_num+1, 1) / self.config.steps_num
        # integrand_val = .5 * (nn_estimate_reweighted_killing(jnp.array(key_stock_y_hat), time_steps, f_trajs, blur_statuses(f_statuses)) + nn_estimate_reweighted_killing(jnp.array(key_stock_y_hat), time_steps, b_trajs, blur_statuses(b_statuses)))
        integrand_val = nn_estimate_reweighted_killing(jnp.array(key_stock_y_hat), time_steps, f_trajs, blur_statuses(f_statuses))

        # TODO: Debug. The formula changes, if mass_0 != 1 or mass_0 < mass_1
        guessed_psi = (1-self.config.mass_1/self.config.mass_0) / jnp.clip(jnp.sum(jnp.sum(integrand_val, axis=1), axis=0) / self.config.steps_num, self.config.eps)

        # TODO: Impove this by turning psi into a scalar
        return key, jnp.array([guessed_psi])