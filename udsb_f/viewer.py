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

from experiment import *


class Viewer():
    """ Object performing sampling and visualization
    """

    def __init__(self, key, experiment: Experiment) -> None:
        self.experiment = experiment
        self.key = key

    def eval_score(self):
        """ Get ready-to-use score networks
        """
        params = self.experiment.get_params()
        return broadcast(lambda model, params: partial(self.experiment.score(model), params), self.experiment.model, params)

    def eval_ferryman(self, direction):
        """ Get ready-to-use Ferryman network
        """
        params = self.experiment.get_params(FERRYMAN)
        return partial(self.experiment.ferryman.apply, params=params, direction=direction)

    def get_fresh_marginal(self, direction):
        """ Sample end marginal
        """
        key = self.key

        key, key_marginal = random.split(key)
        x_marginal = self.experiment.start_marginals_sampler[direction](key_marginal)

        self.key = key

        return x_marginal


    def get_fresh_trajectories(self, direction):
        """ Sample a batch of trajectories
        """
        key = self.key
        key, key_t_init, key_traj = random.split(key, 3)
        self.key = key

        sde = self.experiment.sde

        x_init = self.experiment.start_marginals_sampler[direction](key_t_init)
        trajs, _, statuses = sde.sample_trajectory(key_traj, direction, x_init, self.eval_score(), self.eval_ferryman(direction), corrector="")

        return trajs, _, statuses

    def draw_extreme_marginals(self):
        """ Draw both end marginals
        """
        key = self.key

        key, key_perm = random.split(key)

        self.key = key

        project = self.experiment.e.project

        X_0 = self.get_fresh_marginal(FORWARD)
        X_1 = self.get_fresh_marginal(BACKWARD)

        extreme_projs = jnp.concatenate([project(X_0), project(X_1)])
        extreme_colors = jnp.concatenate([jnp.repeat(jnp.array(plt.cm.tab20(1))[jnp.newaxis,:], X_0.shape[0], axis=0), jnp.repeat(jnp.array(plt.cm.tab20(3))[jnp.newaxis,:], X_1.shape[0], axis=0)])

        shuffling_idxs = random.permutation(key_perm, X_0.shape[0] + X_1.shape[0])

        plt.scatter(*extreme_projs.at[shuffling_idxs].get().T, c=extreme_colors.at[shuffling_idxs].get(), s=1000, alpha=.05)


    def draw_reference_trajectories(self):
        """ Draw trajectories sampled from the prior process
        """
        key = self.key

        sde = self.experiment.sde
        project = self.experiment.e.project

        key, key_init_pnts, key_trajs = random.split(key, 3)
        trajs, _, _ = sde.sample_f_trajectory(key_trajs, self.experiment.e.pi_0_sample(key_init_pnts), broadcast(lambda _: lambda rng, t, x: jnp.zeros_like(x), directions), lambda rng, t: 0., corrector="")

        plt.plot(*jnp.einsum("tnd->dtn", project(trajs.at[::20].get())), color="lightgray", zorder=0);
        plt.scatter(*project(trajs.at[0].get()).T);
        plt.scatter(*project(trajs.at[-1].get()).T);
        plt.axis("equal");

        self.key = key

    def interactive_f_gradient_flow(self):
        """ Interactive visualization of the gradient at a given timestep
        """
        key = self.key

        sde = self.experiment.sde
        project = self.experiment.e.project

        steps_num = self.experiment.e.steps_num
        killing_function_name = self.experiment.e.killing_function_name

        gradient_flow_trajs, _, gradient_flow_statuses = self.get_fresh_trajectories(FORWARD)

        key, key_g_flow = random.split(key)

        self.key = key

        @ipywidgets.interact(probe_time=ipywidgets.FloatSlider(value=.3, min=0., max=1., step=.05))
        def gradient_flow(probe_time):

            state_space_cover = sample_state_space(key_g_flow, gradient_flow_trajs, gradient_flow_statuses, jnp.round(probe_time*steps_num).astype(int))
            
            score_x, score_y = project(sde.f(probe_time, state_space_cover) + sde.g(probe_time, state_space_cover) * self.eval_score()[FORWARD](key_g_flow, probe_time, state_space_cover)).T

            self.draw_extreme_marginals()

            draw_patches(load_patches(killing_function_name), time=None, fig_handle=(plt.gcf(), plt.gca()))
            plt.quiver(project(state_space_cover)[:,0], project(state_space_cover)[:,1], score_x, score_y)

            plt.text(.98, .95, rf"t={probe_time:.2f}", transform=plt.gca().transAxes, ha="right")

            plt.axis("equal");

        return gradient_flow


    def draw_trajectories(self, direction, samples_num=300, timesteps_num=10):
        """ Sample and draw trajectories
        """
        key = self.key
        key, key_chosen_times, key_chosen_particles = random.split(key, 3)
        self.key = key

        project = self.experiment.e.project

        state_dims = self.experiment.e.state_dims
        batch_size = self.experiment.e.batch_size
        steps_num = self.experiment.e.steps_num
        killing_function_name = self.experiment.e.killing_function_name
        
        # TODO: Test more extensively the impact of Langevin correction
        hdim_synth_trajs, _, synth_statuses = self.get_fresh_trajectories(direction)

        assert samples_num < batch_size
        chosen_times = jnp.round(jnp.linspace(0, steps_num, timesteps_num, endpoint=True)).astype(int)
        chosen_particles = random.permutation(key_chosen_particles, hdim_synth_trajs.shape[1])[:samples_num]

        chosen_idxs = jnp.ones(hdim_synth_trajs.shape[:-1]) * 2
        chosen_idxs = chosen_idxs.at[chosen_times].set(chosen_idxs.at[chosen_times].get()-1)
        chosen_idxs = chosen_idxs.at[:,chosen_particles].set(chosen_idxs.at[:,chosen_particles].get()-1)
        chosen_idxs = (chosen_idxs == 0)
        hdim_synth_trajs, synth_statuses = hdim_synth_trajs.at[chosen_idxs].get().reshape((timesteps_num, samples_num, state_dims)), synth_statuses.at[chosen_idxs].get().reshape((timesteps_num, samples_num))
        
        synth_trajs = project(hdim_synth_trajs)

        death_frame = jnp.concatenate([jnp.zeros((1, synth_statuses.shape[1])).astype(bool), jnp.logical_not(synth_statuses.at[1:].get()) * (synth_statuses.at[:-1].get())], axis=0)
        birth_frame = jnp.concatenate([jnp.zeros((1, synth_statuses.shape[1])).astype(bool), (synth_statuses.at[1:].get()) * jnp.logical_not(synth_statuses.at[:-1].get())], axis=0)

        plt.plot(*jnp.einsum("tnd->dtn", synth_trajs), lw=.5, color="lightgray", alpha=.3, zorder=0)
        
        for t in range(1, steps_num, 2):
            plt.scatter(*synth_trajs.at[t,synth_statuses.at[t].get()].get().T, s=.1, color=plt.cm.RdBu_r(t))
            # plt.scatter(*synth_trajs.at[t,synth_statuses.at[t].get() * (hdim_synth_trajs.at[t,:,0].get() == 1.)].get().T, s=.1, color=plt.cm.RdBu_r(t))
            # plt.scatter(*synth_trajs.at[t,synth_statuses.at[t].get() * (hdim_synth_trajs.at[t,:,0].get() == 0.)].get().T, s=.1, color=plt.cm.RdBu(t))

        plt.scatter(*synth_trajs.at[0,synth_statuses.at[0].get()].get().T, s=4, zorder=1, alpha=.7, color="darkblue")
        plt.scatter(*synth_trajs.at[-1,synth_statuses.at[-1].get()].get().T, s=4, zorder=1, alpha=.7, color=plt.cm.tab10(1))
        
        # Plot change of status
        plt.scatter(*synth_trajs.at[death_frame].get().T, s=8, color="black", zorder=1, alpha=.7)
        plt.scatter(*synth_trajs.at[birth_frame].get().T, s=8, color=plt.cm.tab10(3), zorder=1, alpha=.7)
        
        draw_patches(load_patches(killing_function_name), time=None, fig_handle=(plt.gcf(), plt.gca()))
        
        plt.title(f"Direction: {direction}")
        plt.axis("equal");
        plt.show();

        plt.pcolormesh(synth_statuses.T.astype(int), cmap=plt.cm.Greys_r)
        plt.xlabel("Time")
        plt.ylabel("Particle #")

        print(f"Dead particles. End={1-synth_statuses[-1].mean():.3f} \t Min={1-synth_statuses.mean(axis=1).max():.3f} \t Max={1-synth_statuses.mean(axis=1).min():.3f}")

        return hdim_synth_trajs, _, synth_statuses


    def compare_with_baselines(self):
        """ Compare quality of predicted end-marginal against 1st and 2nd order approximations
        """

        self.draw_extreme_marginals()

        project = self.experiment.e.project

        trajs, _, statuses = self.get_fresh_trajectories(FORWARD)

        X_0 = self.get_fresh_marginal(FORWARD)
        X_1 = self.get_fresh_marginal(BACKWARD)

        mean_0 = trajs.at[0, statuses.at[0].get()].get().mean(axis=0)
        std_0 = trajs.at[0, statuses.at[0].get()].get().std(axis=0)
        mean_1 = X_1.mean(axis=0)
        std_1 = X_1.std(axis=0)

        match_mean = trajs.at[0, statuses.at[0].get()].get() + (mean_1 - mean_0)
        match_2nd_moment = std_1 / std_0 * (trajs.at[0, statuses.at[0].get()].get() - mean_0) + mean_1

        # plt.scatter(*project(tmp_f_trajs.at[-1].get()).T, marker='X', s=20, zorder=1, alpha=.5, label="SB-all")
        plt.scatter(*project(match_mean).T, marker=4, s=12, zorder=1, alpha=.5, color=plt.cm.tab20(8), label="match 1st moment")
        plt.scatter(*project(match_2nd_moment).T, marker=5, s=12, zorder=1, alpha=.5, color=plt.cm.tab20(4), label="match 1st & 2nd moments")
        plt.scatter(*project(trajs.at[-1,statuses.at[-1].get()].get()).T, marker=6, s=12, zorder=1, alpha=.5, color=plt.cm.tab20(2), label="SB")
        plt.title("SB vs 2nd order moment-matching")
        plt.legend();

        return pd.DataFrame([compute_scalar_mmd(X_1, trajs.at[-1, statuses.at[-1].get()].get()), compute_scalar_mmd(X_1, match_2nd_moment), compute_scalar_mmd(X_1, match_mean)], index=["SB", "2nd_order_approx", "1st_order_approx"], columns=["MMD_distance"]).sort_values(by="MMD_distance")

