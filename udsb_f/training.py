__author__ = "Matteo Pariset"

import jax
from jax import grad, value_and_grad, jit, tree_map # type: ignore
import jax.random as random
import jax.numpy as jnp
from jax.lax import fori_loop

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
    """ Object representing the status of the training process (losses, optimizers, ...).
    """

    def __init__(self, key, experiment: Experiment, finetune=False) -> None:
        sde = experiment.sde
        model = experiment.model
        ferryman = experiment.ferryman

        objective = experiment.e.objective
        state_dims = experiment.e.state_dims
        batch_size = experiment.e.batch_size

        key, keys_init = split_key(key)
        key, key_ferryman = random.split(key)

        if finetune:
            init_params = experiment.get_params()
        else:
            init_params = broadcast(
                lambda key, model: model.init(key, t=jnp.zeros((batch_size, 1)), x=jnp.zeros((batch_size, state_dims))),
                keys_init,
                model
            ) | {
                FERRYMAN: ferryman.init(key_ferryman, t=jnp.zeros((batch_size, 1)), direction=FORWARD)
            }

        ipf_loss = broadcast(
            lambda direction: value_and_grad(experiment.init_ipf_loss(sde, model, direction, objective), argnums=0),
            directions,
        )

        ferryman_loss = value_and_grad(experiment.init_ferryman_loss(sde, ferryman))

        optimizer = broadcast(lambda _: optax.chain(optax.clip(1.0), optax.adamw(learning_rate=1e-3)), models) | {FERRYMAN: optax.chain(optax.clip(1.0), optax.adamw(learning_rate=1e-2))}

        init_opt_state = broadcast(lambda opt, init_params: opt.init(init_params), optimizer, init_params, score_only=False)

        init_psi = jnp.array([0.])

        def _zero_model(t, x):
            return jnp.zeros_like(x)

        # TODO: zero_model still represents the score (and not 𝜑), maybe rename as zero_score
        zero_model = hk.transform(_zero_model)
        _ = zero_model.init(None, t=jnp.zeros((batch_size, 1)), x=jnp.zeros((batch_size, state_dims)))


        self.experiment = experiment


        # Save training entities
        self.ipf_loss = ipf_loss
        self.ferryman_loss = ferryman_loss

        self.zero_model = zero_model

        self.optimizer = optimizer

        # >>>>>>>>> Push training state <<<<<<<<<
        self.experiment.experiment_state = (key, init_params, init_psi)
        self.opt_state = init_opt_state
        # >>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<

    
    def update_psi(self, psi, f_statuses, b_statuses, td_coeff):
        """ Only used by `UDSB-TD`
        """

        dead_backward = 1 - jnp.clip(b_statuses.mean(axis=1), 1/self.experiment.e.batch_size, 1-1/self.experiment.e.batch_size)
        dead_forward = 1 - jnp.clip(f_statuses.mean(axis=1), 1/self.experiment.e.batch_size, 1-1/self.experiment.e.batch_size)
        new_psi = jnp.power((dead_backward/dead_forward).mean(axis=0), 1/10) * (psi * (psi != 0.) + jnp.array([1.]) * (psi == 0))

        new_psi = jnp.clip(new_psi, 1e-20, 1e7) * (td_coeff > 0)

        return new_psi
    

    @partial(jit, static_argnames=['self', 'd', 'is_warmup'])
    def training_step(self, d, key, params, opt_state, psi, logs, is_warmup=False, td_coeff=.001):
        sde = self.experiment.sde
        model = self.experiment.model
        ferryman = self.experiment.ferryman

        score = self.experiment.score

        start_marginals_sampler = self.experiment.start_marginals_sampler

        paths_reuse = self.experiment.e.paths_reuse
        steps_num = self.experiment.e.steps_num


        key[d], key_init_points, key_traj, key_ferryman_loss = random.split(key[d], 4)

        if is_warmup:
            sampling_score = {
                FORWARD: partial(self.zero_model.apply, None),
                BACKWARD: partial(score(model[BACKWARD]), params[BACKWARD])
            }
            td_coeff = 0.
        else:
            sampling_score = broadcast(lambda d, m: partial(score(m), params[d]), directions, model)

        sampling_ferryman = partial(ferryman.apply, params=params[FERRYMAN], direction=reverse(d))

        init_points = start_marginals_sampler[reverse(d)](key_init_points)
        trajs, _, statuses = sde.sample_trajectory(key_traj, reverse(d), init_points, sampling_score, sampling_ferryman)


        # Need for losses which use difference between successive samples (e.g. mean-matching)
        padded_trajs = jnp.concatenate([trajs, trajs.at[-1:].get()], axis=0)  # shape (k+2,n,d)
        padded_statuses = jnp.concatenate([statuses, statuses.at[-1:].get()], axis=0)  # shape (k+2,n)

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

            ipf_step_loss, ipf_step_grad = self.ipf_loss[d](params[d], params[reverse(d)], key_ipf_loss, euler_m_k, pos_k, pos_k_plus_1, padded_statuses, steps_num)

            logs['ipf_loss'] = logs['ipf_loss'] + ipf_step_loss

            loss = ipf_step_loss
            
            logs['loss'] = logs['loss'] + loss

            # Accumulate gradients
            grads = tree_map(lambda g_acc, g_ipf: g_acc+g_ipf, grads, ipf_step_grad)

            return (
                key,
                params,
                opt_state,
                grads,
                logs
            )

        logs['ferryman_loss'] = 0.

        for _ in range(paths_reuse):
            # Reset gradients
            grads = tree_map(lambda w: jnp.zeros_like(w), params[d])

            path_state = (key, params, opt_state, grads, logs)

            key, params, opt_state, grads, logs = fori_loop(0, steps_num+1, _step, path_state)

            # Follow gradients
            updates, opt_state[d] = self.optimizer[d].update(grads, opt_state[d], params[d])        
            new_params = optax.apply_updates(params[d], updates)
            params[d] = ema(params[d], new_params)

            sampling_d = reverse(d)

            if not is_warmup and is_forward(sampling_d):
                # Learn Ferryman
                ferryman_loss, ferryman_grad = self.ferryman_loss(params[FERRYMAN], key_ferryman_loss, sampling_d, trajs, statuses, td_coeff)

                logs['ferryman_loss'] = logs['ferryman_loss'] + ferryman_loss

                ferryman_updates, opt_state[FERRYMAN] = self.optimizer[FERRYMAN].update(ferryman_grad, opt_state[FERRYMAN], params[FERRYMAN])
                new_ferryman_params = optax.apply_updates(params[FERRYMAN], ferryman_updates)
                
                # TODO: Debug. Deactivate ema?
                # params[FERRYMAN] = ema(params[FERRYMAN], new_ferryman_params)
                params[FERRYMAN] = new_ferryman_params

        state = key, params, opt_state, psi, logs

        state[-1]['ipf_loss'] /= (paths_reuse * (steps_num+1))
        state[-1]['ferryman_loss'] /= paths_reuse

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

    @partial(jit, static_argnames=['self', 'direction', 'corrector'])
    def fast_sample_trajectory_evo(self, key, direction, x_init, params, ferryman, corrector=""):
        score = broadcast(lambda d, m: partial(self.experiment.score(m), params[d]), directions, self.experiment.model)

        if is_forward(direction):
            return self.experiment.sde.sample_f_trajectory(key, x_init, score, ferryman, corrector)
        else:
            return self.experiment.sde.sample_b_trajectory(key, x_init, score, ferryman, corrector)

    def finetune_psi(self, key, params, psi):    
        """ Only used by `UDSB-TD`
        """

        desired_deaths = 1 - self.experiment.e.mass_1/self.experiment.e.mass_0

        if desired_deaths == 0:
            return key, psi * 0.
        elif psi.at[0].get() == 0.:
            psi = psi + 1.

        # TODO: Debug. Test more the effect of high psi value thresholds (was 1e5)
        low_psi = jnp.clip(psi / 100, 1e-20, 1e10)
        high_psi = jnp.clip(psi * 100, 1e-20, 1e10)

        key[FORWARD], key_fuzzing = random.split(key[FORWARD])

        remaining_attempts = 5
        averages_num = 3
        while remaining_attempts > 0:
            remaining_attempts -= 1

            fuzzing_factor = random.uniform(key_fuzzing) * .2

            mid_psi = jnp.exp((.5 + fuzzing_factor) * jnp.log(high_psi) + (.5 - fuzzing_factor) * jnp.log(low_psi))

            key[FORWARD], key_init_points = random.split(key[FORWARD])

            init_points = self.experiment.start_marginals_sampler[FORWARD](key_init_points)

            observed_deaths = 0
            for _ in range(averages_num):
                key[FORWARD], key_traj = random.split(key[FORWARD])
                trajs, ys, statuses = self.fast_sample_trajectory_evo(key_traj, FORWARD, init_points, params, mid_psi)
                observed_deaths += 1 - statuses.at[-1].get().mean()
                del trajs
                del ys
                del statuses

            observed_deaths /= averages_num

            print(f"[{observed_deaths:.2f}]", end="")
    
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

            print(f"({mid_psi[0]:.2f})", end="")

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

            # TODO: Change finetune_psi into finetune_ferryman
            # key_train, psi = self.finetune_psi(key_train, params, psi)
            print(f"finetuned_psi = {psi}")
            
        return key, params, opt_state, psi


    def pretrain(self):
        """ Perform a pretraining step and update the training state vector.
        """

        # <<<<<<<<< Pop training state >>>>>>>>>
        key, params, psi = self.experiment.experiment_state
        opt_state = self.opt_state
        # <<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>


        # # TODO: Debug. Validate pretraining epoch behavior
        # key, params, opt_state = self.pretraining_phase(key, params, opt_state)


        # >>>>>>>>> Push training state <<<<<<<<<
        self.experiment.experiment_state = (key, params, psi)
        self.opt_state = opt_state
        # >>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<

        return self


    def train(self, td_schedule=[None]):
        """ Perform `len(td_schedule)` training phases (each consisting of several epochs).

        On each phase, if `td_schedule[i] == 0`, the Ferryman network is **not** updated. All positive values of `td_schedule[i]`
        have the same effect, i.e., SGD is perfomed on the Ferryman loss.
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

