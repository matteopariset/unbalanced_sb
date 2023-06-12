__author__ = "Matteo Pariset"

import jax
import jax.random as random
import jax.numpy as jnp
from jax.tree_util import tree_map

import pandas as pd
import joblib

import os

import abc


class Dataset(abc.ABC):
    """ Describe a reference process together with marginals.

    The reference process is given by the SDE:
      `d_t X_t = f(t, X_t) dt + g(t, X_t) dW_t`

    where:
      - `f()` is the drift
      - `g()` is the diffusivity

    Initial (resp. final) marginal is described by the function `pi_0_sample` (resp. `pi_1_sample`)
    which allows to randomly sample points from it

    """
    def __init__(self, dataset_name, state_dims, killing_function_name) -> None:
        self.dataset_name = dataset_name
        self.state_dims = state_dims
        self.killing_function_name = killing_function_name

    def export(self):
        return {
            "dataset_name": self.dataset_name,
            "state_dims": self.state_dims,
            "killing_function_name": self.killing_function_name,
        }

    @abc.abstractmethod
    def f(self, t, x):
        pass

    @abc.abstractmethod
    def g(self, t, x):
        pass

    # TODO: Must specify in docs that project must work with arrays having shapes: (n,d) and (k+1,n,d)
    @abc.abstractmethod
    def project(self, state):
        pass

    @abc.abstractmethod
    def as_time(self, day):
        pass

    @abc.abstractmethod
    def pi_0_sample(self, key, n_samples=300):
        pass

    @abc.abstractmethod
    def pi_1_sample(self, key, n_samples=300):
        pass
    
class ToyDataset(Dataset):
    def f(self, t, x):
        return jnp.zeros_like(x)

    def g(self, t, x):
        return 2.

    def project(self, state):
        return state

    def as_time(self, day):
        return (day - 2) / (4 - 2)

    def pi_0_sample(self, key, n_samples=300):
        keys_pi_0 = random.split(key, 3)
        left_line_points = n_samples//5*4
        center_points = n_samples - left_line_points
        return jnp.concatenate([
            jnp.vstack([jnp.zeros(left_line_points) - 5, random.uniform(keys_pi_0[0], (left_line_points,)) * 10 - 5]).T,
            jnp.vstack([random.normal(keys_pi_0[1], shape=(center_points,)) * .5, random.normal(keys_pi_0[2], shape=(center_points,)) * .5]).T
        ])

    def pi_1_sample(self, key, n_samples=300):
        keys_pi_1 = random.split(key)
        return jnp.vstack([jnp.zeros(n_samples) + 5., random.uniform(keys_pi_1[0], (n_samples,)) * 2 - 5 + (random.uniform(keys_pi_1[1], (n_samples,)) > .5) * 8]).T


def triangular_diffusivity(t, max_g=2, min_g=.1):
    return max_g - (max_g-min_g) * 2*jnp.abs(t-.5)


class OutlierWeakDataset(Dataset):
    def f(self, t, x):
        return jnp.zeros_like(x)

    def g(self, t, x):
        return 2.

    def project(self, state):
        return state

    def as_time(self, day):
        return day

    def pi_0_sample(self, key, n_samples=300):
        key_loc, key_group = random.split(key, 2)
        group_assignment = random.choice(key_group, jnp.array([-1,0,1]), shape=(n_samples,))
        return jnp.vstack([jnp.zeros(n_samples) - 5, random.uniform(key_loc, (n_samples,)) * 3 - group_assignment * (6.) - (1-group_assignment) * 1.5]).T
    
    def pi_1_sample(self, key, n_samples=300):
        key_loc, key_group = random.split(key, 2)
        group_assignment = random.choice(key_group, jnp.array([-4, 0,1,2,3, 4,5,6,7])//4, shape=(n_samples,))
        return jnp.vstack([jnp.zeros(n_samples) + 5., random.uniform(key_loc, (n_samples,)) * 3 - group_assignment * (6.) - (1-group_assignment) * 1.5]).T
    

def outlier_death(t, pos):
    dead_zone_mask = (-2 < pos[:,0]) * (pos[:,0] < -1) * (2. <= pos[:,1]) * (pos[:,1] < 6)

    return dead_zone_mask * .5


cells_dataset_name_prefix = "full-4i"
cells_dataset_folder = "../data/4i"

cells_control_name = "control"
cells_drug_name = "ulixertinib"

cells_measurements = pd.read_csv(os.path.join(cells_dataset_folder, f"{cells_dataset_name_prefix}_normalized.csv"))

def cells_time(hours):
    return (hours - 8) / (48 - 8)

class SingleCellDataset(Dataset):
    def __init__(self, dataset_name, state_dims, killing_function_name) -> None:
        super().__init__(dataset_name, state_dims, killing_function_name)

        self.measurements = cells_measurements
        self.feature_names = self.measurements.columns[1:-1]

        timeframes = {
            8:  self.measurements.query("time == 8"),
            24: self.measurements.query("time == 24"),
            48: self.measurements.query("time == 48"),
        }

        # self.timeframe_control = tree_map(lambda tframe: jnp.array(tframe.loc[tframe["Condition"] == cells_control_name, self.feature_names].to_numpy().astype(float)), timeframes)
        self.timeframe_drug = tree_map(lambda tframe: jnp.array(tframe.loc[tframe["Condition"] == cells_drug_name, self.feature_names].to_numpy().astype(float)), timeframes)

        self.proj = joblib.load(f"../data/4i/{cells_dataset_name_prefix}_pca.pkl")

    def as_time(self, hours):
        return cells_time(hours)

    def f(self, t, x):
        return jnp.zeros_like(x)
        # return (t < self.as_time(24)) * (self.timeframe_drug[24].mean(axis=0) - self.timeframe_drug[8].mean(axis=0)) + (self.as_time(24) <= t) * (t < self.as_time(48)) * (self.timeframe_drug[48].mean(axis=0) - self.timeframe_drug[24].mean(axis=0))

    def g(self, t, x):
        # return 2.
        return triangular_diffusivity(t, 8)
        # return 2 * ((t < self.as_time(24)) * (self.timeframe_drug[24].std(axis=0) / self.timeframe_drug[8].std(axis=0)) + (self.as_time(24) <= t) * (t < self.as_time(48)) * (self.timeframe_drug[48].std(axis=0) / self.timeframe_drug[24].std(axis=0)))

    def project(self, state):
        if len(state.shape) == 3:
            return jnp.array(self.proj.transform(state.reshape((-1, state.shape[-1])))).reshape(state.shape[:-1] + (2,))
        elif len(state.shape) == 2:
            return jnp.array(self.proj.transform(state))
        elif len(state.shape) == 1:
            return jnp.array(self.proj.transform(jnp.expand_dims(state, 0))).squeeze()
        
        raise ValueError(f"Could not apply projection. Invalid shape: {state.shape}")

    def pi_0_sample(self, key, n_samples=300):
        pi_0_source = self.timeframe_drug[8]
        tot_samples = pi_0_source.shape[0]

        ret_idxs = random.permutation(key, (n_samples//tot_samples + 1) * tot_samples)[:n_samples]
        return pi_0_source[ret_idxs]
    
    def pi_1_sample(self, key, n_samples=300):
        pi_1_source = self.timeframe_drug[48]
        tot_samples = pi_1_source.shape[0]

        ret_idxs = random.permutation(key, (n_samples//tot_samples + 1) * tot_samples)[:n_samples]
        return pi_1_source[ret_idxs]


def ulixertinib_cells_uniform_killer(t, pos):
    return (cells_time(24) <= t) * jnp.ones(pos.shape[:-1]) * 1e-8

cells_measurements_8h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 8").loc[:,cells_measurements.columns[1:-1]].to_numpy())
means_8h = cells_measurements_8h.mean(axis=0, keepdims=True)
stds_8h = cells_measurements_8h.std(axis=0, keepdims=True)

cells_measurements_24h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 24").loc[:,cells_measurements.columns[1:-1]].to_numpy())
means_24h = cells_measurements_24h.mean(axis=0, keepdims=True)
stds_24h = cells_measurements_24h.std(axis=0, keepdims=True)

cells_measurements_48h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 48").loc[:,cells_measurements.columns[1:-1]].to_numpy())
means_48h = cells_measurements_48h.mean(axis=0, keepdims=True)
stds_48h = cells_measurements_48h.std(axis=0, keepdims=True)

def checkpoint_8h_24h(t, x, std_threshold=2., cutoff=.2):
    effective_t = (t - cells_time(8)) / (cells_time(24) - cells_time(8))
    effective_mean = effective_t * means_24h + (1-effective_t) * means_8h
    effective_std = effective_t * stds_24h + (1-effective_t) * stds_8h
    threshold_violation_score = jnp.mean(jnp.abs(x - effective_mean) > std_threshold * effective_std, axis=1)
    return (threshold_violation_score < cutoff) * 0. + (threshold_violation_score >= cutoff) * threshold_violation_score

def checkpoint_24h_48h(t, x, std_threshold=2., cutoff=.2):
    effective_t = (t - cells_time(24)) / (cells_time(48) - cells_time(24))
    effective_mean = effective_t * means_48h + (1-effective_t) * means_24h
    effective_std = effective_t * stds_48h + (1-effective_t) * stds_24h
    threshold_violation_score = jnp.mean(jnp.abs(x - effective_mean) > std_threshold * effective_std, axis=1)
    return (threshold_violation_score < cutoff) * 0. + (threshold_violation_score >= cutoff) * threshold_violation_score * effective_t

def ulixertinib_cells_measurement_killer(t, pos):
    return (cells_time(8) <= t) * (t < cells_time(24)) * checkpoint_8h_24h(t, pos) * 1e-6 + (cells_time(24) <= t) * checkpoint_24h_48h(t, pos) * 1e-2


class CovidDataset(Dataset):
    variants_fracs_sum = 4

    def __init__(self, dataset_name, state_dims, killing_function_name) -> None:
        super().__init__(dataset_name, state_dims, killing_function_name)

        dataset_folder = "../data/covid/"
        dataset_version_name = os.path.join(dataset_folder, "covid_v3")

        self.t0_points = jnp.load(f"{dataset_version_name}_t0_points.npy")
        self.t1_points = jnp.load(f"{dataset_version_name}_t1_points.npy")

    def f(self, t, x):
        return jnp.zeros_like(x)

    def g(self, t, x):
        return triangular_diffusivity(t, 3)

    def project(self, state):
        return state[...,:2]

    def as_time(self, day):
        return day

    def pi_0_sample(self, key, n_samples=300):
        return self.t0_points.at[random.permutation(key, self.t0_points.shape[0])[:n_samples]].get()

    def pi_1_sample(self, key, n_samples=300):
        return self.t1_points.at[random.permutation(key, self.t1_points.shape[0])[:n_samples]].get()


class CovidDatasetAppearance(Dataset):
    variants_fracs_sum = 4

    def __init__(self, dataset_name, state_dims, killing_function_name) -> None:
        super().__init__(dataset_name, state_dims, killing_function_name)

        dataset_folder = "../data/covid/"
        dataset_version_name = os.path.join(dataset_folder, "covid_v3")

        # Swap endpoints
        self.t0_points = jnp.load(f"{dataset_version_name}_t1_points.npy")
        self.t1_points = jnp.load(f"{dataset_version_name}_t0_points.npy")

    def f(self, t, x):
        # return jnp.ones_like(x) * (self.t1_points[...,:2].mean(axis=0, keepdims=True) - self.t0_points[...,:2].mean(axis=0, keepdims=True)) * .1
        # return jnp.concatenate([jnp.ones(x.shape[:-1] + (2,)) * (self.t1_points[...,:2].mean(axis=0, keepdims=True) - self.t0_points[...,:2].mean(axis=0, keepdims=True)), jnp.zeros(x.shape[:-1] + (x.shape[-1]-2,))], axis=-1) * .1
        drift = jnp.ones_like(x)
        drift = drift.at[...,:2].set((self.t1_points[...,:2].mean(axis=0, keepdims=True) - self.t0_points[...,:2].mean(axis=0, keepdims=True)))
        drift = drift.at[...,2:].set(0)
        return drift

    def g(self, t, x):
        return triangular_diffusivity(t, 3) * jnp.array([.8, .8, 1., 1., 1., 1.])

    def project(self, state):
        return state[...,:2]

    def as_time(self, day):
        return day

    def pi_0_sample(self, key, n_samples=300):
        return self.t0_points.at[random.permutation(key, self.t0_points.shape[0])[:n_samples]].get()

    def pi_1_sample(self, key, n_samples=300):
        return self.t1_points.at[random.permutation(key, self.t1_points.shape[0])[:n_samples]].get()
    


## Killing functions

# checkpoint_3d = as_checkpoint(timeframe[3])

# def neurips22_citeseq(t, pos):
#     checkpoint_3d_test = (as_time(2.5) <= t) * (t < as_time(3)) * checkpoint_3d(pos)

#     # TODO: Debug. Define strength of killing zone
#     return checkpoint_3d_test * .01

def no_killing(t, pos):
    return jnp.zeros(pos.shape[:-1]).astype(bool)

def centered_rectangle_dead_pool(t, pos):
    dead_zone_mask = (2 < pos[:,0]) * (pos[:,0] < 3) * (-5 <= pos[:,1]) * (pos[:,1] < 5)

    return dead_zone_mask * .5

def categorical_split_rectangle_simple(t, pos):

    ## Obstacle interaction
    rectangle_mask_low = (pos[:,0] == 1) * (2 < pos[:,1]) * (pos[:,1] < 3) * (-10 <= pos[:,2]) * (pos[:,2] <= -1)
    rectangle_mask_high = (pos[:,0] == 0) * (2 < pos[:,1]) * (pos[:,1] < 3) * (1 <= pos[:,2]) * (pos[:,2] <= 10)

    rectangle_mask = jnp.logical_or(rectangle_mask_low, rectangle_mask_high)
    rectangle_rate = rectangle_mask * 1.

    return rectangle_rate

def variants_evolution_guardrail(t, pos):
    variants_frac = pos[:,2:]

    penalty_discount = 2.
    death_rate = jnp.clip(jnp.square(variants_frac.sum(axis=1) - CovidDataset.variants_fracs_sum)/penalty_discount, 0., 1.)

    return death_rate

covid_india_embs = pd.read_csv("../data/covid/covid_v3_country_embs.csv").set_index("Unnamed: 0").loc['India'].to_numpy().reshape((-1,2))
covid_us_embs = pd.read_csv("../data/covid/covid_v3_country_embs.csv").set_index("Unnamed: 0").loc['United States'].to_numpy().reshape((-1,2))

def variants_evolution_disappearance(t, pos):
    # variants_frac = pos[:,2:]

    # penalty_discount = 2.
    # death_rate = jnp.clip(jnp.square(variants_frac.sum(axis=1) - CovidDataset.variants_fracs_sum)/penalty_discount, 0., 1.)

    positions = pos[:,:2]
    delta_frac = pos[:,4]

    india_delta_disappearance = (t > .8) * (1-jnp.clip(jnp.abs(positions - covid_india_embs).mean(axis=1), 0., .5)) * (delta_frac > (.3 * CovidDataset.variants_fracs_sum))
    us_delta_disappearance = (t > .55) * (1-jnp.clip(jnp.abs(positions - covid_us_embs).mean(axis=1), 0., .5)) * (delta_frac > (.1 * CovidDataset.variants_fracs_sum))

    return (india_delta_disappearance + us_delta_disappearance) * .1 + variants_evolution_guardrail(t, pos)

# Available datasets
datasets = {
    'toy'  : ToyDataset,
    'outlier-weak': OutlierWeakDataset,
    'covid': CovidDataset,
    'covid-appearance': CovidDatasetAppearance,
    'cells': SingleCellDataset,
}

# Available killing functions
killing_funcs = {
    'no_killing':                           no_killing,
    'centered_rectangle_dead_pool':         centered_rectangle_dead_pool,
    'categorical_split_rectangle_simple':   categorical_split_rectangle_simple,

    'outlier_death':                        outlier_death,

    'variants_evolution_guardrail':         variants_evolution_guardrail,
    'variants_evolution_disappearance':     variants_evolution_disappearance,
    'ulixertinib_cells_uniform_killer':     ulixertinib_cells_uniform_killer,
    'ulixertinib_cells_measurement_killer': ulixertinib_cells_measurement_killer,
}