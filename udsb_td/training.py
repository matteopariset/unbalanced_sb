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


from scipy import linalg
import matplotlib as mpl

from utils import *
from datasets import *
from networks import *
from experiment import Experiment


class Trainer():

    def __init__(self, key, experiment: Experiment, init_psi=jnp.array([0.])) -> None:
        sde = experiment.sde
        model = experiment.model

        objective = experiment.e.objective
        state_dims = experiment.e.state_dims
        batch_size = experiment.e.batch_size

        key, keys_init = split_key(key)

        init_params = broadcast(
            lambda key, model: model.init(key, t=jnp.zeros((batch_size, 1)), x=jnp.zeros((batch_size, state_dims))),
            keys_init,
            model
        )

        ipf_loss = broadcast(
            lambda direction: jax.value_and_grad(experiment.init_ipf_loss(sde, model, direction, objective), argnums=0),
            directions,
        )

        td_loss = broadcast(
            lambda direction: jax.value_and_grad(experiment.init_td_loss(sde, model, direction), argnums=0),
            directions,
        )

        optimizer = broadcast(lambda _: optax.chain(optax.clip(1.0), optax.adamw(learning_rate=1e-3)), directions)

        init_opt_state = broadcast(lambda opt, init_params: opt.init(init_params), optimizer, init_params)

        init_psi = jnp.array([0.])

        def _zero_model(t, x):
            return jnp.zeros_like(x)

        # TODO: zero_model still represents the score (and not 𝜑), maybe rename as zero_score
        zero_model = hk.transform(_zero_model)
        _ = zero_model.init(None, t=jnp.zeros((batch_size, 1)), x=jnp.zeros((batch_size, state_dims)))


        self.experiment = experiment


        # Save training entities
        self.ipf_loss = ipf_loss
        self.td_loss = td_loss

        self.zero_model = zero_model

        self.optimizer = optimizer

        # >>>>>>>>> Push training state <<<<<<<<<
        self.experiment.experiment_state = (key, init_params, init_psi)
        self.opt_state = init_opt_state
        # >>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<



    def update_marginal_estimate(self, d, key, params, opt_state):
        pdf = self.experiment.pdf
        model = self.experiment.model

        density = self.experiment.density

        start_marginals_sampler = self.experiment.start_marginals_sampler


        key, key_marginal_0, key_marginal_1, key_model_t0, key_model_t1 = random.split(key, 5)

        X_0 = start_marginals_sampler[FORWARD](key_marginal_0)
        X_1 = start_marginals_sampler[BACKWARD](key_marginal_1)


        def _marginal_loss(p):
            est_t0 = density(model[d])(p, key_model_t0, 0., X_0)
            est_t1 = density(model[d])(p, key_model_t1, 1., X_1)
            return .5 * (
                jnp.abs(.5 * jnp.log(pdf[FORWARD](X_0)) - est_t0) +
                jnp.abs(.5 * jnp.log(pdf[BACKWARD](X_1)) - est_t1)
            ).mean()

        y_grad = jax.grad(_marginal_loss)(params[d])

        # Follow gradients
        updates, opt_state[d] = self.optimizer[d].update(y_grad, opt_state[d], params[d])        
        params[d] = optax.apply_updates(params[d], updates)

        return key, params, opt_state

    
    
    def center_marginals(self, key, params, opt_state):
        """ Make sure that 𝜑, 𝜑_hat have a reasonable shape at the extremes
        """

        def _step(i, args):
            args = self.update_marginal_estimate(FORWARD, *args)
            args = self.update_marginal_estimate(BACKWARD, *args)
            return args

        key, params, opt_state = jax.lax.fori_loop(0, 500, _step, (key, params, opt_state))

        return key, params, opt_state


    # @partial(jax.jit, static_argnames=['self', 'd', 'is_warmup'])
    # def training_step(self, d, key, params, opt_state, psi, logs, is_warmup=False, td_coeff=.001):
    #     sde = self.experiment.sde
    #     model = self.experiment.model

    #     density = self.experiment.density
    #     score = self.experiment.score

    #     start_marginals_sampler = self.experiment.start_marginals_sampler

    #     paths_reuse = self.experiment.e.paths_reuse
    #     steps_num = self.experiment.e.steps_num


    #     key[d], key_init_points, key_traj, key_traj_reverse = random.split(key[d], 4)

    #     # TODO: Debug. Should we use a custom density during pretraining? 0 comes from  0 = log 1 =? log 𝜑
    #     sampling_density = broadcast(lambda dens, p: partial(dens.apply, p), model, params)
    #     if is_warmup:
    #         sampling_score = {
    #             FORWARD: partial(self.zero_model.apply, None),
    #             BACKWARD: partial(score(model[BACKWARD]), params[BACKWARD])
    #         }
    #         td_coeff = 0.
    #     else:
    #         sampling_score = broadcast(lambda d, m: partial(score(m), params[d]), directions, model)

    #     if True:
    #         # Guess value of psi
    #         key, new_psi_trajs = sde.guess_psi_from_trajs(key, start_marginals_sampler, sampling_density, sampling_score, psi)
    #         key, new_psi_space = sde.guess_psi_from_space(key, start_marginals_sampler, sampling_density, sampling_score, psi)

    #         # # Geometric mean of psi estimates
    #         # new_psi = jnp.sqrt(new_psi_trajs*new_psi_space)

    #         # # Arithmetic mean of psi estimates
    #         # new_psi = (new_psi_trajs + new_psi_space) / 2

    #         # # Harmonic mean of psi estimates
    #         # new_psi = 2 * (new_psi_trajs*new_psi_space) / (new_psi_trajs+new_psi_space)

    #         # trajs estimates
    #         new_psi = new_psi_trajs

    #         # # space estimates
    #         # new_psi = new_psi_space


    #         new_psi = new_psi * (td_coeff > 0)

    #         # TODO: Debug. Restore ema of psi
    #         # psi = ema(psi, new_psi)
    #         psi = new_psi


    #     init_points = start_marginals_sampler[reverse(d)](key_init_points)
    #     trajs, _, statuses = sde.sample_trajectory(key_traj, reverse(d), init_points, sampling_density, sampling_score, psi)

    #     init_td_points = start_marginals_sampler[d](key_init_points)
    #     td_trajs, td_ys, td_statuses = sde.sample_trajectory(key_traj, d, init_td_points, sampling_density, sampling_score, psi)

    #     # Need for losses which use difference between successive samples (e.g. mean-matching)
    #     padded_trajs = jnp.concatenate([trajs, trajs.at[-1:].get()], axis=0)  # shape (k+2,n,d)
    #     padded_statuses = jnp.concatenate([statuses, statuses.at[-1:].get()], axis=0)  # shape (k+2,n)

    #     # TODO: Debug. Blurring is problematic when introducing birth by splitting (would include previous location of recycled particles)
    #     # Consider timesteps just after particle death (or before birth) when computing TD loss
    #     td_blurred_statuses = blur_statuses(td_statuses)
    #     # blurred_statuses = td_statuses

    #     def _step(k: int, args):
    #         if not is_forward(d):
    #             k = steps_num - k

    #         key, params, opt_state, grads, logs = args

    #         pos_k, pos_k_plus_1 = padded_trajs.at[k].get(), padded_trajs.at[k+1].get()

    #         key[d], key_ipf_loss, key_td_loss = random.split(key[d], 3)

    #         if is_forward(d):
    #             euler_m_k = k
    #         else:
    #             # When sampling backward, must use k+1 in Euler-Maruyama discretization
    #             euler_m_k = (k+1)

    #         ipf_step_loss, ipf_step_grad = self.ipf_loss[d](params[d], params[reverse(d)], key_ipf_loss, psi, euler_m_k, pos_k, pos_k_plus_1, padded_statuses, steps_num)
    #         # TODO: Refactor this
    #         if is_warmup:
    #             td_step_loss, td_step_grad = 0., jax.tree_map(lambda x: jnp.zeros_like(x), ipf_step_grad)
    #         else:
    #             # TD loss has to be scaled by the number of timesteps used
    #             td_step_loss, td_step_grad = self.td_loss[d](params[d], key_td_loss, psi, k, td_trajs, td_ys, td_blurred_statuses)

    #             td_step_loss = td_coeff * td_step_loss / (steps_num+1)
    #             td_step_grad = jax.tree_map(lambda p: td_coeff * p / (steps_num+1), td_step_grad)

    #         logs['ipf_loss'] = logs['ipf_loss'] + ipf_step_loss
    #         logs['td_loss'] = logs['td_loss'] + td_step_loss

    #         loss = ipf_step_loss + td_step_loss
            
    #         logs['loss'] = logs['loss'] + loss

    #         # Accumulate gradients
    #         grads = jax.tree_map(lambda g_acc, g_ipf, g_td: g_acc+g_ipf+g_td, grads, ipf_step_grad, td_step_grad)

    #         return (
    #             key,
    #             params,
    #             opt_state,
    #             grads,
    #             logs
    #         )

    #     for _ in range(paths_reuse):

    #         # Reset gradients
    #         grads = jax.tree_map(lambda w: jnp.zeros_like(w), params[d])

    #         # Same direction sampling for TD loss
    #         key[d], key_td = random.split(key[d])

    #         td_density = broadcast(lambda m, p: partial(density(m), p), model, params)
    #         td_score = broadcast(lambda m, p: partial(score(m), p), model, params)

    #         # td_ys = sde.sample_y_given_trajectory(key_td, d, td_density, td_score, psi, td_trajs, td_statuses)

    #         path_state = (key, params, opt_state, grads, logs)

    #         key, params, opt_state, grads, logs = jax.lax.fori_loop(0, steps_num+1, _step, path_state)

    #         # Follow gradients
    #         updates, opt_state[d] = self.optimizer[d].update(grads, opt_state[d], params[d])        
    #         new_params = optax.apply_updates(params[d], updates)
    #         params[d] = ema(params[d], new_params)

    #     # if is_forward(d):
    #     #     # Restore original computation of psi + b_centering
    #     #     dead_backward = 1 - jnp.clip(statuses.sum(axis=1) / statuses.at[0].get().sum(), 1/self.experiment.e.batch_size, 1-1/self.experiment.e.batch_size)
    #     #     dead_forward = 1 - jnp.clip(td_statuses.mean(axis=1), 1/self.experiment.e.batch_size, 1-1/self.experiment.e.batch_size)
    #     #     new_psi = jnp.power((dead_backward/dead_forward).mean(axis=0), 1/10) * (psi * (psi != 0.) + jnp.array([1.]) * (psi == 0))

    #     #     new_psi = jnp.clip(new_psi, 1e-20, 1e3) * (td_coeff > 0)

    #     #     psi = new_psi

    #     state = key, params, opt_state, psi, logs

    #     state[-1]['ipf_loss'] /= (paths_reuse * (steps_num+1))
    #     state[-1]['td_loss'] /= paths_reuse

    #     state[-1]['loss'] /= (paths_reuse * (steps_num+1))
        
    #     return state


    @partial(jax.jit, static_argnames=['self', 'd', 'is_warmup'])
    def training_step(self, d, key, params, opt_state, psi, logs, is_warmup=False, td_coeff=.001):
        sde = self.experiment.sde
        model = self.experiment.model

        density = self.experiment.density
        score = self.experiment.score

        start_marginals_sampler = self.experiment.start_marginals_sampler

        paths_reuse = self.experiment.e.paths_reuse
        steps_num = self.experiment.e.steps_num


        key[d], key_init_points, key_traj, key_traj_reverse = random.split(key[d], 4)

        # TODO: Debug. Should we use a custom density during pretraining? 0 comes from  0 = log 1 =? log 𝜑
        sampling_density = broadcast(lambda dens, p: partial(dens.apply, p), model, params)
        if is_warmup:
            sampling_score = {
                FORWARD: partial(self.zero_model.apply, None),
                BACKWARD: partial(score(model[BACKWARD]), params[BACKWARD])
            }
            td_coeff = 0.
        else:
            sampling_score = broadcast(lambda d, m: partial(score(m), params[d]), directions, model)

        # if is_forward(d):
        #     # Guess value of psi
        #     key, new_psi_trajs = sde.guess_psi_from_trajs(key, start_marginals_sampler, sampling_density, sampling_score, psi)
        #     key, new_psi_space = sde.guess_psi_from_space(key, start_marginals_sampler, sampling_density, sampling_score, psi)

        #     # # Geometric mean of psi estimates
        #     # new_psi = jnp.sqrt(new_psi_trajs*new_psi_space)

        #     # # Arithmetic mean of psi estimates
        #     # new_psi = (new_psi_trajs + new_psi_space) / 2

        #     # # Harmonic mean of psi estimates
        #     # new_psi = 2 * (new_psi_trajs*new_psi_space) / (new_psi_trajs+new_psi_space)

        #     # TODO: Debug. Use this for psi-fixed
        #     # trajs estimates
        #     new_psi = new_psi_trajs

        #     # # space estimates
        #     # new_psi = new_psi_space


        #     new_psi = new_psi * (td_coeff > 0)

        #     # TODO: Debug. Restore ema of psi
        #     # psi = ema(psi, new_psi)
        #     psi = new_psi

        init_points = start_marginals_sampler[reverse(d)](key_init_points)
        trajs, ys_reverse, statuses = sde.sample_trajectory_evo(key_traj, reverse(d), init_points, sampling_density, sampling_score, psi)

        # init_td_points = start_marginals_sampler[d](key_init_points)
        # td_trajs, td_ys, td_statuses = sde.sample_trajectory(key_traj, d, init_td_points, sampling_density, sampling_score, psi)

        # Need for losses which use difference between successive samples (e.g. mean-matching)
        padded_trajs = jnp.concatenate([trajs, trajs.at[-1:].get()], axis=0)  # shape (k+2,n,d)
        padded_statuses = jnp.concatenate([statuses, statuses.at[-1:].get()], axis=0)  # shape (k+2,n)

        # TODO: Debug. Blurring is problematic when introducing birth by splitting (would include previous location of recycled particles)
        # Consider timesteps just after particle death (or before birth) when computing TD loss
        td_blurred_statuses = blur_statuses(statuses)
        # blurred_statuses = td_statuses
        
        if is_forward(d):
            init_td_points = start_marginals_sampler[d](key_init_points)
            _, _, td_statuses = sde.sample_trajectory(key_traj, d, init_td_points, sampling_density, sampling_score, psi)
            # Restore original computation of psi + b_centering
            dead_backward = 1 - jnp.clip(statuses.sum(axis=1) / statuses.at[0].get().sum(), 1/self.experiment.e.batch_size, 1-1/self.experiment.e.batch_size)
            dead_forward = 1 - jnp.clip(td_statuses.mean(axis=1), 1/self.experiment.e.batch_size, 1-1/self.experiment.e.batch_size)
            new_psi = jnp.power((dead_backward/dead_forward).mean(axis=0), 1/10) * (psi * (psi != 0.) + jnp.array([1.]) * (psi == 0))

            new_psi = jnp.clip(new_psi, 1e-20, 1e7) * (td_coeff > 0)

            psi = new_psi


        def _step(k: int, args):
            if not is_forward(d):
                k = steps_num - k

            key, params, opt_state, grads, logs = args

            pos_k, pos_k_plus_1 = padded_trajs.at[k].get(), padded_trajs.at[k+1].get()

            key[d], key_ipf_loss, key_td_loss = random.split(key[d], 3)

            if is_forward(d):
                euler_m_k = k
            else:
                # When sampling backward, must use k+1 in Euler-Maruyama discretization
                euler_m_k = (k+1)

            ipf_step_loss, ipf_step_grad = self.ipf_loss[d](params[d], params[reverse(d)], key_ipf_loss, psi, euler_m_k, pos_k, pos_k_plus_1, padded_statuses, steps_num)
            # TODO: Refactor this
            if is_warmup:
                td_step_loss, td_step_grad = 0., jax.tree_map(lambda x: jnp.zeros_like(x), ipf_step_grad)
            else:
                # TD loss has to be scaled by the number of timesteps used
                td_step_loss, td_step_grad = self.td_loss[d](params[d], key_td_loss, psi, k, trajs, ys_reverse, td_blurred_statuses)

                td_step_loss = td_coeff * td_step_loss / (steps_num+1)
                td_step_grad = jax.tree_map(lambda p: td_coeff * p / (steps_num+1), td_step_grad)

            logs['ipf_loss'] = logs['ipf_loss'] + ipf_step_loss
            logs['td_loss'] = logs['td_loss'] + td_step_loss

            loss = ipf_step_loss + td_step_loss
            
            logs['loss'] = logs['loss'] + loss

            # Accumulate gradients
            grads = jax.tree_map(lambda g_acc, g_ipf, g_td: g_acc+g_ipf+g_td, grads, ipf_step_grad, td_step_grad)

            return (
                key,
                params,
                opt_state,
                grads,
                logs
            )

        for _ in range(paths_reuse):

            # Reset gradients
            grads = jax.tree_map(lambda w: jnp.zeros_like(w), params[d])

            # Same direction sampling for TD loss
            key[d], key_td = random.split(key[d])

            td_density = broadcast(lambda m, p: partial(density(m), p), model, params)
            td_score = broadcast(lambda m, p: partial(score(m), p), model, params)

            # td_ys = sde.sample_y_given_trajectory(key_td, d, td_density, td_score, psi, td_trajs, td_statuses)

            path_state = (key, params, opt_state, grads, logs)

            key, params, opt_state, grads, logs = jax.lax.fori_loop(0, steps_num+1, _step, path_state)

            # Follow gradients
            updates, opt_state[d] = self.optimizer[d].update(grads, opt_state[d], params[d])        
            new_params = optax.apply_updates(params[d], updates)
            params[d] = ema(params[d], new_params)

        state = key, params, opt_state, psi, logs

        state[-1]['ipf_loss'] /= (paths_reuse * (steps_num+1))
        state[-1]['td_loss'] /= paths_reuse

        state[-1]['loss'] /= (paths_reuse * (steps_num+1))
        
        return state

        

    def pretraining_phase(self, key, init_params, init_opt_state):

        params = init_params

        key, key_train = split_key(key)
        # TODO: Should replace forward params with None, to make sure that in the future they don't get accidentally modified
        # TODO: Should offer non-JITed version of training step
        # # params = init_params.copy()
        _, params, opt_state, _, logs = self.training_step(BACKWARD, key_train, params, init_opt_state, psi=jnp.array([0.]), logs=init_logs(epoch=-1), is_warmup=True, td_coeff=0.)
        print_logs(logs)

        return key, params, opt_state

    @partial(jax.jit, static_argnames=['self', 'direction', 'corrector'])
    def fast_sample_trajectory_evo(self, key, direction, x_init, params, psi, corrector=""):
        density = broadcast(lambda d, m: partial(self.experiment.density(m), params[d]), directions, self.experiment.model)
        score = broadcast(lambda d, m: partial(self.experiment.score(m), params[d]), directions, self.experiment.model)

        if is_forward(direction):
            return self.experiment.sde.sample_f_trajectory_evo(key, x_init, density, score, psi, corrector)
        else:
            return self.experiment.sde.sample_b_trajectory_evo(key, x_init, density, score, psi, corrector)

    def finetune_psi(self, key, params, psi):        
        # TODO: Generalize this to births
        desired_deaths = 1 - self.experiment.e.mass_1/self.experiment.e.mass_0

        if desired_deaths == 0:
            return key, psi * 0.
        elif psi.at[0].get() == 0.:
            psi = psi + 1.

        # TODO: Debug. Test more the effect of high psi value thresholds (was 1e5)
        low_psi = jnp.clip(psi / 100, 1e-20, 1e10)
        high_psi = jnp.clip(psi * 100, 1e-20, 1e10)

        remaining_attempts = 5
        while remaining_attempts > 0:
            remaining_attempts -= 1

            mid_psi = jnp.exp((jnp.log(high_psi) + jnp.log(low_psi)) / 2)

            key[FORWARD], key_init_points, key_traj = random.split(key[FORWARD], 3)

            init_points = self.experiment.start_marginals_sampler[FORWARD](key_init_points)
            trajs, ys, statuses = self.fast_sample_trajectory_evo(key_traj, FORWARD, init_points, params, mid_psi)
            observed_deaths = 1 - statuses.at[-1].get().mean()

            del trajs
            del ys
            del statuses
    
            if jnp.isclose(observed_deaths, desired_deaths, rtol=0, atol=.05):
                # Set +/-5% threshold of identity
                print(f"=")
                return key, mid_psi
            
            elif observed_deaths < desired_deaths:
                print(f"^", end="")
                low_psi = mid_psi
    
            else:
                print(f"v", end="")
                high_psi = mid_psi

        print("X: selecting default")
        default_psi = jnp.exp((jnp.log(high_psi) + jnp.log(low_psi)) / 2)
        return key, default_psi

    def training_phase(self, key, params, opt_state, psi, td_coeff=None, epchs_num=5):
        key, key_train = split_key(key)

        for epch in tqdm(range(epchs_num)):

            if td_coeff is None:
                # TODO: Debug. Maybe it's too much
                epch_td_coeff = 1. * epch/epchs_num
            else:
                epch_td_coeff = td_coeff

            # TODO: Refactor this + killing
            for _ in range(10):
                key_train, params, opt_state, psi, logs = self.training_step(FORWARD, key_train, params, opt_state, psi, init_logs(epoch=epch), td_coeff=epch_td_coeff)
                print_logs(logs)
                print(f"raw_psi = {psi}")
            for _ in range(10):
                key_train, params, opt_state, psi, logs = self.training_step(BACKWARD, key_train, params, opt_state, psi, init_logs(epoch=epch), td_coeff=epch_td_coeff)
                print_logs(logs)
                print(f"psi = {psi}")

            key_train, psi = self.finetune_psi(key_train, params, psi)
            print(f"finetuned_psi = {psi}")
            
        return key, params, opt_state, psi


    def pretrain(self):
        """ Perform a pretraining step and update the training state vector
        """

        # <<<<<<<<< Pop training state >>>>>>>>>
        key, params, psi = self.experiment.experiment_state
        opt_state = self.opt_state
        # <<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>


        # Prepare log-density at marginals
        key, params, opt_state = self.center_marginals(key, params, opt_state)

        # # TODO: Debug. Validate pretraining epoch behavior
        # key, params, opt_state = self.pretraining_phase(key, params, opt_state)


        # >>>>>>>>> Push training state <<<<<<<<<
        self.experiment.experiment_state = (key, params, psi)
        self.opt_state = opt_state
        # >>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<

        return self


    def train(self, td_schedule=[None]):
        """ Perform `len(td_schedule)` training phases (each consisting of several epochs) using `td_schedule[i]` as TD coefficient during phase `i`
        """

        # <<<<<<<<< Pop training state >>>>>>>>>
        key, params, psi = self.experiment.experiment_state
        opt_state = self.opt_state
        # <<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>


        for phase_num in range(len(td_schedule)):
            td_coeff = td_schedule[phase_num]

            key, params, opt_state, psi = self.training_phase(key, params, opt_state, psi, td_coeff=td_coeff)


        # >>>>>>>>> Push training state <<<<<<<<<
        self.experiment.experiment_state = (key, params, psi)
        self.opt_state = opt_state
        # >>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<

        # Perform automatic saving
        self.experiment.save()

        return self

