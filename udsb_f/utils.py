__author__ = "Matteo Pariset"

import jax
import jax.random as random
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jax import vmap # type: ignore

from sklearn.metrics.pairwise import rbf_kernel

import os

from scipy import linalg
import matplotlib as mpl
import itertools
import numpy

import json


## Globals

directions = {"forward":"forward", "backward":"backward"}
models = directions | {"ferryman": "ferryman"}

# Representations of directions
FORWARD = directions["forward"]
BACKWARD = directions["backward"]
FERRYMAN = models["ferryman"]

## Utils

def info(*msg):
    print("[INFO]:", *msg)

def is_forward(direction):
    """
    Safe test of direction: returns True, if `direction` == FORWARD
    but raises an exception if `direction` has an invalid value
    """
    if direction == FORWARD:
        return True
    elif direction == BACKWARD:
        return False
    else:
        raise ValueError(f"Unknown direction: {direction}")

def reverse(direction):
    if is_forward(direction):
        return BACKWARD
    else:
        return FORWARD

def broadcast(f, *args, score_only=True):
    if score_only:
        models_list = directions
    else:
        models_list = models

    return {k: f(*[arg[k] for arg in args]) for k in models_list.values()}

def split_key(key):
    key, *rs = random.split(key, 3)
    return key, {
        FORWARD: rs[0],
        BACKWARD: rs[1]
    }

def blur_statuses(statuses):
    return jnp.logical_or(
        jnp.concatenate([statuses.at[1:].get(), statuses.at[-1:].get()], axis=0), 
        jnp.logical_or(
            jnp.concatenate([statuses.at[:1].get(), statuses.at[:-1].get()], axis=0),
            statuses
        )
    )

def gaussian_kernel(var_coeff=2):
    kde_kernel = vmap(
        lambda pos, x: 1/jnp.sqrt(jnp.power(2*jnp.pi*var_coeff, x.shape[-1])) * jnp.exp(-jnp.sum(jnp.square(jnp.expand_dims(x, 0) - pos), axis=-1)/(2*var_coeff**2)),
        in_axes=(None, 0)
    )

    return kde_kernel

def kde(kernel, x_ref, x):
    return kernel(x_ref, x).mean(axis=1)


### Visualization

def sample_state_space(key, trajs, statuses, k):
    points = trajs.at[k, statuses.at[k].get()].get()
    stds = trajs.at[k, statuses.at[k].get()].get().std(axis=0)

    grid_shape = (100, points.shape[1])

    points = points[:grid_shape[0],:] + random.normal(key, (grid_shape[1],)) * .05 * stds

    return points

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])
def plot_results(X, Y_, means, covariances, title, project=lambda x: x, fig_handle=None):
    X = project(X)
    means = project(means)

    if fig_handle is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_handle

    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        w = project(w.T).T
        v = 2.0 * numpy.sqrt(2.0) * numpy.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not numpy.any(Y_ == i):
            continue
        ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, alpha=.4, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = numpy.arctan(u[1] / u[0])
        angle = 180.0 * angle / numpy.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_alpha(0.5)
        ax.add_patch(ell)

        # ax.scatter(mean[0], mean[1], s=800, color=color, alpha=.8, zorder=10)

    plt.title(title)

    return fig, ax

def figure(fig_handle):
    if fig_handle is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_handle

    return fig, ax

def link_dynamic_times(attr, time):
    if isinstance(attr, (list, tuple)):
        return [link_dynamic_times(a, time) for a in attr]
    elif isinstance(attr, str) and attr.find("@t") > -1:
        assert time is not None, "Must specify time value"
        return eval(attr.replace("@t", "t"), None, {"t": time})
    else:
        return attr

# TODO: Sync implementations with create_datasets (draw_patches has been modified here!!!!!!)
def draw_patches(patches, time=None, fig_handle=None):
    fig, ax = figure(fig_handle)
    common_styles = {
        'hatch': '/' * 8,
        'ec': 'lightgray',
        'fc': [0,0,0,0],
        'clip_on': False,
    }
    # Remove existing patches
    for old_p in ax.patches:
        old_p.remove()
    
    # Add new ones
    for p, all_args in patches:
        patch_style = common_styles.copy()
        patch_style.update(map(lambda it: (it[0], link_dynamic_times(it[1], time)), all_args[1].items()))
        
        ax.add_patch(p(*all_args[0], **patch_style))
        ax.set_aspect('equal', adjustable='datalim')
    return fig, ax

patches_root_dir = "../data/2d/"

def load_patches(dataset_name):
    with open(os.path.join(patches_root_dir, f"{dataset_name}-patches.json"), "r") as f:
        return list(map(lambda p: (getattr(plt, p[0]), *p[1:]), json.loads(f.read())))



        
### Reproducibility

def get_config_file(dataset_name, tag):
    config_file = f"./configs/{dataset_name}__{tag}.json"
    return config_file

def get_params_file(dataset_name, tag):
    params_file = f"./weights/{dataset_name}__{tag}.npz"
    return params_file

def experiment_exists(dataset_name, tag):
    params_file = get_params_file(dataset_name, tag)
    return os.path.exists(params_file)

def save_experiment(dataset_name, tag, key, params, psi):
    if experiment_exists(dataset_name, tag):
        raise ValueError("An run with the same dataset and experiment name already exists!")
    else:
        jnp.savez(get_params_file(dataset_name, tag), key=key, params=params, psi=psi)

def load_experiment(dataset_name, tag):
    if experiment_exists(dataset_name, tag):
        info("Reloading params from cache")
        content = jnp.load(get_params_file(dataset_name, tag), allow_pickle=True)
        key, params, psi = content['key'], content['params'][()], content['psi']
        return jnp.array(key), params, jnp.array(psi)
    else:
        raise ValueError("Experiment not found")

def init_logs(epoch: int):
    return {
        'epoch': epoch,
        'ipf_loss': 0.,
        'td_loss': 0.,
        'ferryman_loss': 0.,
        'loss': 0.,
    }

def print_logs(logs):
    print(f"EPOCH #{logs['epoch']} \t loss={logs['loss']:.3f} \t ipf_loss={logs['ipf_loss']:.3f} \t ferryman_loss={logs['ferryman_loss']:.3f}")



### Miscellaneous

def as_checkpoint(tframe, std_threshold=1.5, agreement_threshold=.90):
    empirical_mean = tframe.mean(axis=0, keepdims=True)
    empirical_std = tframe.std(axis=0, keepdims=True)
    def _chkpt(x):
        return jnp.mean(jnp.abs(x - empirical_mean) <= std_threshold * empirical_std, axis=1) < agreement_threshold
    return _chkpt

def ema(old_params, new_params, decay=0.99):
    return jax.tree_map(lambda old, new: (1-decay)*old + decay*new, old_params, new_params)

def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()

def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = jnp.nan
        return mmd

    return jnp.mean(jnp.array(list(map(lambda x: safe_mmd(target, transport, x), gammas))))



## Computational resources

def as_jax_pdf(gm, total_mass=1.):
    means, covs, weights = jnp.array(gm.means_), jnp.array(gm.covariances_), jnp.array(gm.weights_)  # shape (components_num,)

    def eval_cdf(x):  # shape (n,d)
        # TODO: Debug. Should convert -inf into (custom) neginf?
        return jnp.nan_to_num(
            jax.vmap(
                jax.vmap(
                    lambda x, mean, cov, weight: jax.scipy.stats.multivariate_normal.pdf(x, mean, cov) * weight,
                    in_axes=(0, None, None, None), out_axes=(0)
                ),
                in_axes=(None, 0, 0, 0), out_axes=(1)
            )(x, means, covs, weights).sum(axis=1) * total_mass
        )

    return eval_cdf

def divergence(key, F, t, x):
    """ Estimate the divergence of `F` using random (+/- 1) projections.

    For `(a_i) ~ unif{-1,+1}`, the function `Q` defined by:
    
     `Q = a · ∇(a·F)`
    
    is an unbiased estimator of `∇·F`, i.e.

     `∇·F = E[Q]`
    """

    n = x.shape[0]
    # Repetitions to compute empirical expectation
    r = 20
    d = x.shape[1]

    rand_a = random.rademacher(key, (n, r, d))

    def _gaussian_proj(a, t, x):
        F_of_t_x = F(t, jnp.expand_dims(x, axis=0))[0]   # shape: (d,)
        return jnp.inner(a, F_of_t_x)
    grad = jax.grad(_gaussian_proj, argnums=2) # shape: (d,)

    assert len(x.shape) == 2, x.shape

    vec_grad = jax.vmap(jax.vmap(grad, in_axes=(0, None, 0), out_axes=0), in_axes=(1, None, None), out_axes=1)(rand_a, t, x)  # shape: (n, r, d)

    Q = jnp.sum(rand_a * vec_grad, axis=2)  # shape: (n, r)

    return jnp.sum(Q, axis=1) / r  # shape: (n,)

def match_locations_with_candidates(locs_mask, cands_mask):
    """ Returns an array of indices, where the `i`-th component points to the location `j` to associate to candidate `i`

    **WARNING**: The indices associated to non-candidates are set to `0`: should therefore filter by `cands_mask` after applying the matches
    """
    locs_idxs = jnp.where(locs_mask, jnp.arange(locs_mask.shape[0]), locs_mask.shape[0])
    cands_idxs = jnp.where(cands_mask, jnp.arange(cands_mask.shape[0]), cands_mask.shape[0])

    # Must extend conditions to avoid out-of-bounds indices
    locs_idxs = jnp.concatenate([locs_idxs, jnp.array([locs_mask.shape[0]])])
    cands_idxs = jnp.concatenate([cands_idxs, jnp.array([cands_mask.shape[0]])])
    
    matches = jnp.sort(locs_idxs).at[jnp.argsort(jnp.argsort(cands_idxs))].get().at[:-1].get()

    # Replace placeholder indices with index 0 (leaving invalid indices could be catastrophic in Jax)
    placeholders_mask = (matches == locs_mask.shape[0])
    matches = jnp.logical_not(placeholders_mask) * matches + placeholders_mask * 0

    return matches

def birth_by_splitting(key, status, birth_mask, reference_pos):
    key, key_half_splitting = random.split(key)
    # # TODO: Test this, need to re-establish symmetry between rate of birth (by splitting) and death
    # birth_mask = birth_mask * (jax.random.uniform(key_half_splitting) < .5)

    birth_locations_mask = jnp.logical_and(status, birth_mask)

    dead_trajs_count = jnp.cumsum(jnp.logical_not(status).astype(int))
    # TODO: Shuffle trajectories to resurrect (otherwise there may be bias coming from the description of the marginal sampler, e.g. rectangle_*)
    candidate_particles_mask = jnp.logical_and(jnp.logical_not(status), dead_trajs_count <= birth_locations_mask.astype(int).sum())

    matches = match_locations_with_candidates(birth_locations_mask, candidate_particles_mask)

    extended_candidate_parts_mask = jnp.expand_dims(candidate_particles_mask, axis=-1)
    next_pos = jnp.logical_not(extended_candidate_parts_mask) * reference_pos + extended_candidate_parts_mask * reference_pos.at[matches].get()

    curr_status = jnp.logical_or(status, candidate_particles_mask)
    
    return key, curr_status, next_pos