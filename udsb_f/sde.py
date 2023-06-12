__author__ = "Matteo Pariset"

import jax
from jax import grad, vmap # type: ignore
import jax.random as random
import jax.numpy as jnp
from jax.lax import fori_loop

import os
import sys

from scipy import linalg
import matplotlib as mpl

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

    def __init__(self, config, f, g, pdf, killer, splitting_births_frac) -> None:
        self.config = config
        self.f = f
        self.g = g
        self.pdf = pdf

        # Gaussian KDE
        kde_var_coeff = 2
        self.kde_kernel = gaussian_kernel(kde_var_coeff)

        # TODO: Support stateful killing functions
        self.killer = killer

        self.splitting_births_frac = splitting_births_frac

        # TODO: Make parameter
        # Available choices: "keep", "freeze"
        self.dead_mode = "keep"

    def get_normalized_mass(self, direction):
        assert len(self.config.times) >= 2
        assert len(self.config.mass) == len(self.config.times)
        assert self.config.times[0] == 0
        assert self.config.times[-1] == self.config.steps_num

        mass_max = max(self.config.mass)

        if is_forward(direction):
            return self.config.mass[0]/mass_max
        else:
            return self.config.mass[-1]/mass_max


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
        
        key, next_pos = fori_loop(0, 20, _langevin_step, (key, pos)) # type: ignore

        return key, next_pos


    def estimate_density(self, ref_pos, status, x):
        density_est = (status * self.kde_kernel(ref_pos, x)).sum(axis=1) / jnp.clip(status.sum(), 1)

        return density_est


    def apply_killer(self, key, death_threshold):
        key, key_deaths, key_births = random.split(key, 3)
        deaths = random.uniform(key_deaths, (death_threshold.shape[0],)) < death_threshold
        births = -random.uniform(key_births, (death_threshold.shape[0],)) < -1-death_threshold

        return key, deaths, births

    def reweight_killing(self, key, direction, death_threshold, ferryman, f_ref_pos, status, t, pos, clip=True):
        """ Computes the posterior death/birth probabilities starting from the prior ones, in `death_threshold`.
        """

        # Non-parametric density estimation
        f_density_est = self.estimate_density(f_ref_pos, status, pos)

        key, key_ferryman = random.split(key)

        # Add 1 to the density to avoid unreachability walls at the beginning of training
        if is_forward(direction):
            death_reweighting = ferryman(rng=key_ferryman, t=t) / (1. + self.config.eps + f_density_est)
        else:
            K_t = jnp.clip(1-status.mean(), 1/self.config.batch_size)
            death_reweighting = ferryman(rng=key_ferryman, t=t) / (1. + self.config.eps + f_density_est)

        death_threshold = death_threshold * death_reweighting

        if clip:
            death_threshold = jnp.clip(death_threshold, -1., 1.)

        return key, death_threshold

    def match_locations_with_candidates(self, locs_mask, cands_mask):
        """ Returns an array of indices, where the `i`-th component points to the location `j` to associate to candidate `i`

        **WARNING**: The indices associated to non-candidates are set to `0`: should therefore filter by `cands_mask` after applying the matches
        """
        locs_idxs = jnp.where(locs_mask, jnp.arange(locs_mask.shape[0]), locs_mask.shape[0])
        cands_idxs = jnp.where(cands_mask, jnp.arange(cands_mask.shape[0]), cands_mask.shape[0])

        # Must extend conditions to avoid out-of-bounds indices
        locs_idxs = jnp.concatenate([locs_idxs, jnp.array([locs_mask.shape[0]])]) # type: ignore
        cands_idxs = jnp.concatenate([cands_idxs, jnp.array([cands_mask.shape[0]])]) # type: ignore
        
        matches = jnp.sort(locs_idxs).at[jnp.argsort(jnp.argsort(cands_idxs))].get().at[:-1].get()

        # Replace placeholder indices with index 0 (leaving invalid indices could be catastrophic in Jax)
        placeholders_mask = (matches == locs_mask.shape[0])
        matches = jnp.logical_not(placeholders_mask) * matches + placeholders_mask * 0

        return matches


    def choose_birth_type(self, key, splitting_frac, birth_mask):
        from_splitting = jnp.logical_and((random.uniform(key, birth_mask.shape) < splitting_frac), birth_mask)
        from_shadow_paths = jnp.logical_and(jnp.logical_not(from_splitting), birth_mask)

        return from_shadow_paths, from_splitting

    def birth_from_shadow_paths(self, status, birth_mask):
        reborn = jnp.logical_and(jnp.logical_not(status), birth_mask)
        status = jnp.logical_or(status, reborn)

        return status

    
    def birth_by_splitting(self, key, status, birth_mask, reference_pos):
        key, key_half_splitting = random.split(key)

        birth_locations_mask = jnp.logical_and(status, birth_mask)

        dead_trajs_count = jnp.cumsum(jnp.logical_not(status).astype(int))
        # TODO: Shuffle trajectories to resurrect (otherwise there may be bias coming from the description of the marginal sampler, e.g. rectangle_*)
        # Ensure same number of locations & birth candidates 
        candidate_particles_mask = jnp.logical_and(jnp.logical_not(status), dead_trajs_count <= birth_locations_mask.astype(int).sum())
        birth_locations_mask = jnp.logical_and(birth_locations_mask, jnp.cumsum(birth_locations_mask.astype(int)) <= candidate_particles_mask.astype(int).sum())

        matches = self.match_locations_with_candidates(birth_locations_mask, candidate_particles_mask)

        extended_candidate_parts_mask = jnp.expand_dims(candidate_particles_mask, axis=-1)
        next_pos = jnp.logical_not(extended_candidate_parts_mask) * reference_pos + extended_candidate_parts_mask * reference_pos.at[matches].get()

        curr_status = jnp.logical_or(status, candidate_particles_mask)
        
        return key, curr_status, next_pos


    def sample_f_trajectory(self, key, x_0, score, ferryman, corrector):
        steps_num = self.config.steps_num

        traj = jnp.zeros((steps_num+1, *x_0.shape))
        traj = traj.at[0].set(x_0)

        key, key_init_status = random.split(key)

        statuses = jnp.zeros(traj.shape[:-1]).astype(bool)

        # Start with right proportion of dead/alive particles
        statuses = statuses.at[0].set(random.uniform(key_init_status, shape=(x_0.shape[0],)) < self.get_normalized_mass(FORWARD))

        key, key_density_b_init = random.split(key)

        def _step(i, args):
            key, curr_pos, part_traj, part_statuses = args

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

            if corrector == "langevin":
                key, next_pos = self.langevin_correct(key, score, t, next_pos)

            if self.killer is not None:
                death_threshold = self.killer(t, next_pos)
                key, death_threshold = self.reweight_killing(key, FORWARD, death_threshold, ferryman, curr_pos, part_statuses.at[i-1].get(), t, next_pos)
                
                key, dead_mask, birth_mask = self.apply_killer(key, death_threshold)

                curr_status = part_statuses.at[i-1].get()

                # DEATHS ######################
                curr_status = jnp.logical_and(curr_status, jnp.logical_not(dead_mask))
                ###############################
                
                # BIRTHS ######################
                key, key_birth_type = random.split(key)
                shadow_births_mask, splitting_births_mask = self.choose_birth_type(key_birth_type, self.splitting_births_frac, birth_mask)

                # ...from shadow paths
                curr_status = self.birth_from_shadow_paths(curr_status, shadow_births_mask)
                # ...from splitting
                key, curr_status, next_pos = self.birth_by_splitting(key, curr_status, splitting_births_mask, next_pos)
                ###############################

                part_statuses = part_statuses.at[i].set(curr_status)

                if self.dead_mode == "keep":
                    pass
                elif self.dead_mode == "freeze":
                    raise NotImplementedError("Cannot freeze trajectories")

            part_traj = part_traj.at[i].set(next_pos)

            return key, next_pos, part_traj, part_statuses

        key, x_1, traj, statuses = fori_loop(1, steps_num+1, _step, (key, x_0, traj, statuses)) # type: ignore

        return traj, None, statuses


    def sample_b_trajectory(self, key, x_1, score, ferryman, corrector):
        steps_num = self.config.steps_num

        traj = jnp.zeros((steps_num+1, *x_1.shape))
        traj = traj.at[-1].set(x_1)

        key, key_init_status = random.split(key)

        statuses = jnp.zeros(traj.shape[:-1]).astype(bool)  # shape (k+1,n)
        # Start with right proportion of dead/alive particles
        statuses = statuses.at[-1].set(random.uniform(key_init_status, shape=(x_1.shape[0],)) < self.get_normalized_mass(BACKWARD))

        birth_places = jnp.copy(x_1)  # shape (n,d)

        key, key_density_f_init = random.split(key)

        def _step(i, args):
            key, curr_pos, part_traj, part_statuses, part_birth_places = args
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

            if corrector == "langevin":
                key, next_pos = self.langevin_correct(key, score, t, next_pos)

            if self.killer is not None:
                death_threshold = self.killer(t, next_pos)
                key, death_threshold = self.reweight_killing(key, BACKWARD, death_threshold, ferryman, curr_pos, part_statuses.at[i+1].get(), t, next_pos)

                key, birth_mask, dead_mask = self.apply_killer(key, death_threshold)

                curr_status = part_statuses.at[i+1].get()

                # DEATHS ######################
                curr_status = jnp.logical_and(curr_status, jnp.logical_not(dead_mask))
                ###############################
                
                # BIRTHS ######################
                key, key_birth_type = random.split(key)
                shadow_births_mask, splitting_births_mask = self.choose_birth_type(key_birth_type, self.splitting_births_frac, birth_mask)

                # ...from shadow paths
                curr_status = self.birth_from_shadow_paths(curr_status, shadow_births_mask)
                # ...from splitting
                key, curr_status, next_pos = self.birth_by_splitting(key, curr_status, splitting_births_mask, next_pos)
                ###############################

                part_statuses = part_statuses.at[i].set(curr_status)

                born_now = jnp.logical_and(jnp.logical_not(part_statuses.at[i+1].get()), part_statuses.at[i].get())

                born_now_mask = jnp.expand_dims(born_now, axis=1)

                part_birth_places = born_now_mask * next_pos + jnp.logical_not(born_now_mask) * part_birth_places

            part_traj = part_traj.at[i].set(next_pos)

            return key, next_pos, part_traj, part_statuses, part_birth_places

        key, x_0, traj, statuses, _ = fori_loop(1, steps_num+1, _step, (key, x_1, traj, statuses, birth_places)) # type: ignore


        if self.dead_mode == "keep":
            pass
        elif self.dead_mode == "freeze":
            assert False, "Unimplemented"

        return traj, None, statuses


    def sample_trajectory(self, key, direction, x_init, score, ferryman, corrector=""):
        if is_forward(direction):
            return self.sample_f_trajectory(key, x_init, score, ferryman, corrector)
        else:
            return self.sample_b_trajectory(key, x_init, score, ferryman, corrector)
