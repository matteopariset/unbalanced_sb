__author__ = "Matteo Pariset"

# %env XLA_PYTHON_CLIENT_MEM_FRACTION=.40

import pandas as pd

from experiment import *
from training import *
from viewer import Viewer
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

plt.rcParams["figure.figsize"] = (6.4, 4.8)

plt.rcParams['font.family'] = 'serif'
sns.set_context(context='talk', font_scale=.9)
palette = ['#1A254B', '#114083', '#A7BED3', '#F2545B', '#A4243B']

cmap = LinearSegmentedColormap.from_list('cmap', palette, N=18)

colors = ['#1A254B', '#114083', '#A7BED3', '#FFFFFF', '#F2545B', '#A4243B']
from matplotlib.colors import LinearSegmentedColormap
bcmap = LinearSegmentedColormap.from_list('bcmap', colors, N=100)

# Schrodinger Bridges solver

def get_no_outlier_config():
    dataset_config = {
        'dataset_name': "no-outlier",
        'state_dims': 2,
        'killing_function_name': 'outlier_death_birth', # 'centered_rectangle_dead_pool'
    }

    experiment_config = {
        ## Experiment-dependent variables ############################
        'objective': "divergence",

        'times': [0, 50, 100],
        'mass': [1000, None, None],

        'init_components_num': 2,
        'end_components_num': 2,

        'ipf_mask_dead': False,
        'ferryman_layers_num': 5,
        'splitting_births_frac': 0.8,
        'reality_coefficient': .8,

        'steps_num': 100,
        'batch_size': 512,
        'paths_reuse': 5,
        'hidden_dims': 32,

        'eps': 1e-7,
        'neginf': -10,
        'plusinf': 7,

        'experiment_name': f"v4_final",
    }

    config = {
        "dataset": dataset_config,
        "experiment": experiment_config
    }

    return config


def get_death_config():
    dataset_config = {
        'dataset_name': "toy",
        'state_dims': 2,
        'killing_function_name': 'centered_rectangle_dead_pool', # 'centered_rectangle_dead_pool'
    }

    experiment_config = {
        ## Experiment-dependent variables ############################
        'objective': "divergence",

        'times': [0, 100],
        'mass': [1000, None],

        'init_components_num': 2,
        'end_components_num': 2,

        'ipf_mask_dead': True,

        'steps_num': 100,
        'batch_size': 512,
        'paths_reuse': 5,
        'hidden_dims': 32,

        'eps': 1e-7,
        'neginf': -10,
        'plusinf': 7,

        'experiment_name': f"v4_ferryman",
    }

    config = {
        "dataset": dataset_config,
        "experiment": experiment_config
    }

    return config


def get_deflection_config():
    dataset_config = {
        'dataset_name': "toy",
        'state_dims': 2,
        'killing_function_name': 'split_rectangle_dead_pool', # 'rectangle_death_birth', # 'centered_rectangle_dead_pool'
    }

    experiment_config = {
        ## Experiment-dependent variables ############################
        'objective': "mean_matching",

        'times': [0, 100],
        'mass': [1000, None],

        'init_components_num': 2,
        'end_components_num': 2,

        'ipf_mask_dead': True,
        
        'ferryman_activate_final': False,

        'steps_num': 100,
        'batch_size': 512,
        'paths_reuse': 5,
        'hidden_dims': 32,

        'eps': 1e-7,
        'neginf': -10,
        'plusinf': 7,

        'experiment_name': f"v4_readying_figures",
    }

    config = {
        "dataset": dataset_config,
        "experiment": experiment_config
    }

    return config



def get_death_birth_config():
    dataset_config = {
        'dataset_name': "toy",
        'state_dims': 2,
        'killing_function_name': 'rectangle_death_birth', # 'centered_rectangle_dead_pool'
    }

    experiment_config = {
        ## Experiment-dependent variables ############################
        'objective': "divergence",

        'times': [0, 50, 100],
        'mass': [1000, None, None],

        'init_components_num': 2,
        'end_components_num': 2,

        'ipf_mask_dead': False,

        'ferryman_layers_num': 5,

        'steps_num': 100,
        'batch_size': 512,
        'paths_reuse': 5,
        'hidden_dims': 32,

        'eps': 1e-7,
        'neginf': -10,
        'plusinf': 7,

        'experiment_name': f"v4_ferryman",
    }

    config = {
        "dataset": dataset_config,
        "experiment": experiment_config
    }

    return config


def run_no_outlier_experiment():
    # Define mass constraints
    middle_mass = jnp.linspace(630, 1000, 6)
    end_mass = jnp.linspace(750, 1300, 6)

    # Load config
    config = get_no_outlier_config()

    # Store statistics
    obtained_middle = []
    obtained_end = []

    for mm, em in zip(middle_mass, end_mass):
        print(f"1000 -> {mm} -> {em}")

        config['experiment']['mass'][-2] = int(mm)
        config['experiment']['mass'][-1] = int(em)

        experiment = Experiment.create(config)
        trainer = Trainer(random.PRNGKey(0), experiment)

        trainer.train([1., 1., 1.])
        experiment.save(f"v4_div_final_large_no_mask-{int(mm)}-{int(em)}")

        viewer = Viewer(random.PRNGKey(0), experiment)
        _, _, sim_statuses = viewer.get_fresh_trajectories(FORWARD)
        obtained_middle.append(sim_statuses[50].mean())
        obtained_end.append(sim_statuses[100].mean())

    print("desired mm:", middle_mass)
    print("obtained mm:", jnp.array(obtained_middle))

    print("desired em:", end_mass)
    print("obtained em:", jnp.array(obtained_end))


def run_death_experiment():
    # Define mass constraints
    end_mass = jnp.linspace(450, 1000, 6)

    # Load config
    config = get_death_config()

    # Store statistics
    obtained_end = []

    for em in end_mass:
        print(f"1000 -> {em}")

        config['experiment']['mass'][-1] = int(em)

        experiment = Experiment.create(config)
        trainer = Trainer(random.PRNGKey(0), experiment)

        trainer.train([1., 1., 1.])
        experiment.save(f"v4_final_death_mask_mass-{int(em)}")

        viewer = Viewer(random.PRNGKey(0), experiment)
        _, _, sim_statuses = viewer.get_fresh_trajectories(FORWARD)
        obtained_end.append(sim_statuses[100].mean())

    print("desired em:", end_mass)
    print("obtained em:", jnp.array(obtained_end))


def run_deflection_experiment():
    # Define mass constraints
    end_mass = jnp.linspace(300, 1000., 6)[::-1]

    # Load config
    config = get_deflection_config()

    # Store statistics
    obtained_end = []

    for em in end_mass:
        print(f"1000 -> {em}")

        config['experiment']['mass'][-1] = int(em)

        experiment = Experiment.create(config)
        trainer = Trainer(random.PRNGKey(0), experiment)

        trainer.train([1., 1., 1.])
        experiment.save(f"v4_final_splitting_mm_deflection_centered_killer_mass-{int(em)}")

        viewer = Viewer(random.PRNGKey(0), experiment)
        _, _, sim_statuses = viewer.get_fresh_trajectories(FORWARD)
        obtained_end.append(sim_statuses[100].mean())

    print("desired em:", end_mass)
    print("obtained em:", jnp.array(obtained_end))


def run_death_birth_experiment():
    # Define mass constraints
    middle_mass = jnp.linspace(550, 1000, 6)
    end_mass = jnp.linspace(750, 1300, 6)

    # Load config
    config = get_death_birth_config()

    # Store statistics
    obtained_middle = []
    obtained_end = []

    for mm, em in zip(middle_mass, end_mass):
        print(f"1000 -> {mm} -> {em}")

        config['experiment']['mass'][-2] = int(mm)
        config['experiment']['mass'][-1] = int(em)

        experiment = Experiment.create(config)
        trainer = Trainer(random.PRNGKey(0), experiment)

        trainer.train([1., 1., 1.])
        experiment.save(f"v4_final_large_death-births_no-mask_mass-{int(mm)}-{int(em)}")

        viewer = Viewer(random.PRNGKey(0), experiment)
        _, _, sim_statuses = viewer.get_fresh_trajectories(FORWARD)
        obtained_middle.append(sim_statuses[50].mean())
        obtained_end.append(sim_statuses[100].mean())

    print("desired mm:", middle_mass)
    print("obtained mm:", jnp.array(obtained_middle))

    print("desired em:", end_mass)
    print("obtained em:", jnp.array(obtained_end))



run_no_outlier_experiment()
# run_death_experiment()
# run_deflection_experiment()
# run_death_birth_experiment()
