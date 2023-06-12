__author__ = "Matteo Pariset"

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import jax
import jax.numpy as jnp
import jax.random as random
import ott

import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from experiment import *
from viewer import Viewer
from utils import *

plt.rcParams["figure.figsize"] = (6.4, 4.8)

plt.rcParams['font.family'] = 'serif'
sns.set_context(context='talk', font_scale=.9)
palette = ['#1A254B', '#114083', '#A7BED3', '#F2545B', '#A4243B']

cmap = LinearSegmentedColormap.from_list('cmap', palette, N=18)

colors = ['#1A254B', '#114083', '#A7BED3', '#FFFFFF', '#F2545B', '#A4243B']
from matplotlib.colors import LinearSegmentedColormap
bcmap = LinearSegmentedColormap.from_list('bcmap', colors, N=100)


def plot_marginal(marginal, dim_reduction, **kwargs):
    plt.scatter(*dim_reduction(marginal).T, **kwargs)
    plt.axis("equal")
    
def plot_marginals(t0_marginal, t1_marginal, t0_color=cmap(.2), t1_color=cmap(.8), projection=lambda x: x, **kwargs):
    if 'alpha' not in kwargs:
        kwargs['alpha'] = .4
    plt.scatter(*projection(t0_marginal).T, color=t0_color, label=r"$t_0$", **kwargs)
    plt.scatter(*projection(t1_marginal).T, color=t1_color, label=r"$t_1$", **kwargs)
    plt.axis("equal")
    plt.legend();

def plot_multiple_marginals(marginals, statuses, dim_reduction, step_nums, normalize=False, labels=None, s=.5):
    if normalize:
        marginals = norm_step(marginals)
    for i, step in enumerate(step_nums):
        if isinstance(labels, bool) and not labels:
            label = None
        elif labels is not None and len(labels) > 2:
            label = labels[i]
        elif labels is not None and len(labels) == 2:
            if step == step_nums[0]:
                label = labels[0]
            elif step == step_nums[-1]:
                label = labels[1]
            else:
                label = None
        else:
            label = None
        plot_marginal(marginals[i,statuses[i]], dim_reduction, s=s, color=cmap(i/float(marginals.shape[0]-1)), label=label)
    plt.legend()

def plot_matchings(t0_points, t1_points, projection=lambda x: x, **kwargs):
    kwargs["color"] = kwargs.get("color", "gray")
    kwargs["alpha"] = kwargs.get("alpha", .7)
    kwargs["lw"] = kwargs.get("lw", .2)
    extended_coords = np.concatenate([projection(t0_points), projection(t1_points)], axis=1)
    plt.plot(extended_coords[:,::2].T, extended_coords[:,1::2].T, zorder=0, **kwargs);

def plot_predictions(preds, projection=lambda x: x):
    plt.scatter(*projection(preds[-1]).T, s=7.5, zorder=1, edgecolors="black", facecolors="black", label="predictions")


def draw_particle_paths(viewer, trajs=None):
    if trajs is None:
        trajs, _, statuses = viewer.get_fresh_trajectories(FORWARD)

    fig, ax = plt.subplots()

    selected_trajs = random.choice(viewer.key, trajs.shape[1], (5,))

    fig.set_size_inches(26, 15)
    plts = [
        ax.plot(*jnp.einsum("ijk->kij", trajs), color="grey", lw=.09)[0],
        ax.plot(*jnp.einsum("ijk->kij", trajs[:,selected_trajs]), color="black", lw=1)[0]
    ]

def export_fig(fig_name, force=True, extension="svg"):
    fig_path = f"../figures/{fig_name}.{extension}"
    if not os.path.exists(fig_path) or force:
        plt.savefig(fig_path, bbox_inches="tight", transparent=True)

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
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))


def square_norm(vec):
    return np.square(vec).sum(axis=-1)

def compute_rmsd(target, predicted):
    return np.sqrt(square_norm(target-predicted).mean(axis=0))

def compute_norm_rmsd(target, predicted):
    return compute_rmsd(target, predicted) / (np.sqrt(square_norm(target).max()) - np.sqrt(square_norm(target).min()))

# def compute_l2_difference(target, predicted):
#     return np.sqrt(square_norm(target-predicted)).mean(axis=0)

def compute_l2_difference(target, predicted):
    return np.sqrt(square_norm(target.mean(axis=0) - predicted.mean(axis=0)))


def make_geometry(t0_points, t1_points):
    """ Set up inital/final cloud points living in space endowed with squared Eucliden distance
    """
    point_cloud = ott.geometry.pointcloud.PointCloud(t0_points, t1_points, ott.geometry.costs.SqEuclidean())
    return point_cloud
def compute_ot(t0_points, t1_points):
    """ Solve OT problem
    """
    point_cloud = make_geometry(t0_points, t1_points)
    sinkhorn = ott.solvers.linear.sinkhorn.Sinkhorn()(ott.problems.linear.linear_problem.LinearProblem(point_cloud))
    return sinkhorn
def transport(ot, init_points):
    return ot.to_dual_potentials().transport(init_points)
def compute_wasserstein_2(preds, true):
    ot = compute_ot(preds, true)
    return jnp.sqrt(ot.transport_cost_at_geom(make_geometry(preds, true))).item()

def compute_metrics(target, predicted):
    return {
        'mmd': compute_scalar_mmd(target, predicted),
        'norm_rmsd': compute_norm_rmsd(target, predicted).item(),
        'l2': compute_l2_difference(target, predicted).item(),
        'w2': compute_wasserstein_2(target, predicted),
    }

def draw_trajs_with_killing(experiment, **kwargs):
    viewer = Viewer(random.PRNGKey(0), experiment)
    sampled_trajs = [viewer.get_fresh_trajectories(FORWARD) for _ in range(4)]
    trajs, statuses = jnp.concatenate([x[0] for x in sampled_trajs], axis=1), jnp.concatenate([x[2] for x in sampled_trajs], axis=1)

    plot_marginals(
        experiment.start_marginals_sampler[FORWARD](viewer.key)[:300],
        experiment.start_marginals_sampler[BACKWARD](viewer.key)[:300],
        projection=experiment.e.project)
    plot_multiple_marginals(trajs[::12], statuses[::12], experiment.e.project, range(0, 100, 12), labels=False, **kwargs)
    draw_patches(load_patches(experiment.e.killing_function_name), time=None, fig_handle=(plt.gcf(), plt.gca()));

    # Identify deaths
    death_frame = jnp.cumsum(1-statuses, axis=0) == 1
    death_frame = death_frame.at[0].set(False)
    for f_num, d_mask in enumerate(death_frame):
        plot_marginal(trajs.at[f_num, d_mask].get(), dim_reduction=experiment.e.project, s=1., color="black", zorder=10)

    # Identify births
    birth_frame = (jnp.cumsum(1-statuses[::-1], axis=0) == 1)[::-1]
    if experiment.e.splitting_births_frac > 0.:
        birth_frame = jnp.roll(birth_frame, 1, axis=0)
        birth_frame = birth_frame.at[0].set(False)
    else:
        birth_frame = birth_frame.at[-1].set(False)
    for f_num, b_mask in enumerate(birth_frame):
        plot_marginal(trajs.at[f_num, b_mask].get(), dim_reduction=experiment.e.project, s=5., marker="P", color="white", edgecolors="black", linewidths=.3, zorder=10)

    runs = []
    for _ in range(3):
        sampled_trajs = [viewer.get_fresh_trajectories(FORWARD) for _ in range(4)]
        trajs, statuses = jnp.concatenate([x[0] for x in sampled_trajs], axis=1), jnp.concatenate([x[2] for x in sampled_trajs], axis=1)
        samples_num = min(400, statuses[-1].sum())
        runs.append(compute_metrics(viewer.get_fresh_marginal(BACKWARD)[:samples_num], trajs.at[-1, statuses[-1]].get()[:samples_num]))
    
    print(pd.DataFrame(runs).mean())

    plt.axis("off")