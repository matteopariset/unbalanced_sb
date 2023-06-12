__author__ = "Matteo Pariset"

import jax
import jax.random as random
import jax.numpy as jnp
from jax import vmap
from jax.tree_util import tree_map

import pandas as pd
import joblib

import os
import datetime

import abc

from utils import gaussian_kernel, kde


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
        return day

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


class OutlierCompleteDataset(Dataset):
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
        group_assignment = random.choice(key_group, jnp.array([-1,0,1]), shape=(n_samples,))
        return jnp.vstack([jnp.zeros(n_samples) + 5., random.uniform(key_loc, (n_samples,)) * 3 - group_assignment * (6.) - (1-group_assignment) * 1.5]).T
    
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

    
def progressive_death(t, pos):
    dead_zone_mask_1 = (-3.5 < pos[:,0]) * (pos[:,0] < -2.5) * (2. <= pos[:,1]) * (pos[:,1] < 6)
    dead_zone_mask_2 = (-0.5 < pos[:,0]) * (pos[:,0] < 0.5) * (-2. <= pos[:,1]) * (pos[:,1] < 2)
    dead_zone_mask_3 = (2.5 < pos[:,0]) * (pos[:,0] < 3.5) * (-6. <= pos[:,1]) * (pos[:,1] < -2)

    return dead_zone_mask_1 * .5 + dead_zone_mask_2 * .5 + dead_zone_mask_3 * .5


class OutlierDataset(Dataset):
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
        group_assignment = random.choice(key_group, jnp.array([0,1]), shape=(n_samples,))
        return jnp.vstack([jnp.zeros(n_samples) + 5., random.uniform(key_loc, (n_samples,)) * 3 - group_assignment * (6.) - (1-group_assignment) * 1.5]).T
    

def outlier_death(t, pos):
    dead_zone_mask = (-2 < pos[:,0]) * (pos[:,0] < -1) * (2. <= pos[:,1]) * (pos[:,1] < 6)

    return dead_zone_mask * .5


class OutlierBirthDataset(Dataset):
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
        group_assignment = random.choice(key_group, jnp.array([-1,1]), shape=(n_samples,))
        return jnp.vstack([jnp.zeros(n_samples) - 5, random.uniform(key_loc, (n_samples,)) * 3 - group_assignment * (6.) - (1-group_assignment) * 1.5]).T
    
    def pi_1_sample(self, key, n_samples=300):
        key_loc, key_group = random.split(key, 2)
        group_assignment = random.choice(key_group, jnp.array([0,1]), shape=(n_samples,))
        return jnp.vstack([jnp.zeros(n_samples) + 5., random.uniform(key_loc, (n_samples,)) * 3 - group_assignment * (6.) - (1-group_assignment) * 1.5]).T
    

def outlier_death_birth(t, pos):
    dead_zone_mask = (-2 <= pos[:,0]) * (pos[:,0] < -1) * (2. <= pos[:,1]) * (pos[:,1] < 6)
    birth_zone_mask = (1 <= pos[:,0]) * (pos[:,0] < 2) * (-2 <= pos[:,1]) * (pos[:,1] < 2)

    return dead_zone_mask * .5 + birth_zone_mask * (-.5)



class CategoricalDataset(Dataset):
    def f(self, t, x):
        return jnp.zeros_like(x)

    def g(self, t, x):
        return jnp.array([2., 2., 0.])

    def project(self, state):
        return state[...,:2]

    def as_time(self, day):
        return day

    def pi_0_sample(self, key, n_samples=300):
        key_loc, key_cat = random.split(key, 2)
        return jnp.concatenate([
            jnp.vstack([jnp.zeros(n_samples) - 5, random.uniform(key_loc, (n_samples,)) * 10 - 5]).T,
            (random.uniform(key_cat, (n_samples, 1)) < .5).astype(float)
        ], axis=1)

    def pi_1_sample(self, key, n_samples=300):
        keys_pi_1 = random.split(key)
        t1_points = jnp.zeros((n_samples, 3))
        t1_points = t1_points.at[:,:2].set(jnp.vstack([jnp.zeros(n_samples) + 5., random.uniform(keys_pi_1[0], (n_samples,)) * 2 - 5 + (random.uniform(keys_pi_1[1], (n_samples,)) > .5) * 8]).T)
        t1_points = t1_points.at[:,2].set((t1_points[:,1] > 0).astype(float))

        return t1_points


def categorical_split_rectangle(t, pos):

    ## Obstacle interaction
    rectangle_mask_high = (-0.5 < pos[:,0]) * (pos[:,0] < 0.5) * (1 <= pos[:,1]) * (pos[:,1] <= 10) * (pos[:,2] == 1)
    rectangle_mask_low =  (-0.5 < pos[:,0]) * (pos[:,0] < 0.5) * (-10 <= pos[:,1]) * (pos[:,1] <= -1) * (pos[:,2] == 0)

    rectangle_mask = jnp.logical_or(rectangle_mask_low, rectangle_mask_high)
    rectangle_rate = rectangle_mask * jnp.inf

    return rectangle_rate



def triangular_diffusivity(t, max_g=2., min_g=.1):
    return max_g - (max_g-min_g) * 2*jnp.abs(t-.5)


def batch_project(proj):
    def _project(state):
        if len(state.shape) == 3:
            return jnp.array(proj.transform(state.reshape((-1, state.shape[-1])))).reshape(state.shape[:-1] + (2,))
        elif len(state.shape) == 2:
            return jnp.array(proj.transform(state))
        elif len(state.shape) == 1:
            return jnp.array(proj.transform(jnp.expand_dims(state, 0))).squeeze()
        
        raise ValueError(f"Could not apply projection. Invalid shape: {state.shape}")
    
    return _project


def cells_time(hours):
    return (hours - 8) / (48 - 8)

class SingleCellDataset(Dataset):
    @staticmethod
    @abc.abstractmethod
    def get_dataset_version():
        raise NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def get_drug_name():
        raise NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def get_dataset_name():
        raise NotImplementedError
    
    @staticmethod
    @abc.abstractmethod
    def load_measurements(split="train"):
        raise NotImplementedError
    

    def __init__(self, dataset_name, state_dims, killing_function_name) -> None:
        super().__init__(dataset_name, state_dims, killing_function_name)

        self.measurements = self.load_measurements()
        self.feature_names = self.measurements.columns[1:-1]

        timeframes = {
            8:  self.measurements.query("time == 8"),
            24: self.measurements.query("time == 24"),
            48: self.measurements.query("time == 48"),
        }

        self.timeframe_drug = tree_map(lambda tframe: jnp.array(tframe.loc[tframe["Condition"] == self.get_drug_name(), self.feature_names].to_numpy().astype(float)), timeframes)

        self.proj = batch_project(joblib.load(f"{self.get_dataset_name()}_pca.pkl"))

        self.means = tree_map(lambda dframe: dframe.mean(axis=0), self.timeframe_drug)
        self.std = tree_map(lambda dframe: dframe.std(axis=0), self.timeframe_drug)

    def as_time(self, hours):
        return cells_time(hours)
    
    def f(self, t, x):
        return 0.
    
    def g(self, t, x):
        return triangular_diffusivity(1.5, 1.)

    def project(self, state):
        return self.proj(state)

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


class TrametinibErlotinibDataset(SingleCellDataset):

    @staticmethod
    def get_dataset_version():
        dataset_version = "v2"
        return dataset_version
    
    @staticmethod
    def get_drug_name():
        drug_name = "trametinib_erlotinib"
        return drug_name

    @staticmethod
    def get_dataset_name():
        dataset_folder = "../data/4i"
        dataset_name = os.path.join(dataset_folder, f"{TrametinibErlotinibDataset.get_dataset_version()}_full-4i")
        return dataset_name
    
    @staticmethod
    def load_measurements(split="train"):
        idxs_store = jnp.load(f"{TrametinibErlotinibDataset.get_dataset_name()}_train_test_split_idxs.npz")
        
        if split == "train":
            idxs = idxs_store['train_idxs']
        elif split == "test":
            idxs = idxs_store['test_idxs']
        else:
            raise ValueError(f"Unknown split '{split}'")
        
        measurements = pd.read_csv(f"{TrametinibErlotinibDataset.get_dataset_name()}_normalized.csv").iloc[idxs]
        return measurements
    
    @staticmethod
    def build_killer(chk_8_24, chk_24_48):
        def _killer(t, pos):
            # Birth should happen "close" to the spline interpolation
            return (cells_time(8) <= t) * (t < cells_time(24)) * jnp.clip(-1. + chk_8_24(t, pos), None, 0.) * 1e-1 + (cells_time(24) <= t) * chk_24_48(t, pos, std_threshold=1.2) * 2e-1
        return _killer

class UlixertinibDataset(SingleCellDataset):

    @staticmethod
    def get_dataset_version():
        dataset_version = "v4"
        return dataset_version

    @staticmethod
    def get_drug_name():
        drug_name = "ulixertinib"
        return drug_name

    @staticmethod
    def get_dataset_name():
        dataset_folder = "../data/4i"
        dataset_name = os.path.join(dataset_folder, f"{UlixertinibDataset.get_dataset_version()}_full-4i")
        return dataset_name
    
    @staticmethod
    def load_measurements(split="train"):
        idxs_store = jnp.load(f"{UlixertinibDataset.get_dataset_name()}_train_test_split_idxs.npz")
        
        if split == "train":
            idxs = idxs_store['train_idxs']
        elif split == "test":
            idxs = idxs_store['test_idxs']
        else:
            raise ValueError(f"Unknown split '{split}'")
        
        measurements = pd.read_csv(f"{UlixertinibDataset.get_dataset_name()}_normalized.csv").iloc[idxs]
        return measurements
    
    @staticmethod
    def build_killer(chk_8_24, chk_24_48):
        def _killer(t, pos):
            return (cells_time(8) <= t) * (t < cells_time(24)) * chk_8_24(t, pos) * 1e-2 +  (cells_time(24) <= t) * (t < cells_time(48)) * chk_24_48(t, pos) * 1e-1
        return _killer


    def g(self, t, x):
        return triangular_diffusivity(2.5, .35)


def cells_measurement_killer(dataset):
    cells_measurements = dataset.load_measurements()
    cells_drug_name = dataset.get_drug_name()

    cells_measurements_8h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 8").loc[:,cells_measurements.columns[1:-1]].to_numpy()) # type: ignore
    means_8h = cells_measurements_8h.mean(axis=0, keepdims=True)
    stds_8h = cells_measurements_8h.std(axis=0, keepdims=True)

    cells_measurements_24h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 24").loc[:,cells_measurements.columns[1:-1]].to_numpy()) # type: ignore
    means_24h = cells_measurements_24h.mean(axis=0, keepdims=True)
    stds_24h = cells_measurements_24h.std(axis=0, keepdims=True)

    cells_measurements_48h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 48").loc[:,cells_measurements.columns[1:-1]].to_numpy()) # type: ignore
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
    
    return dataset.build_killer(checkpoint_8h_24h, checkpoint_24h_48h)


def ulixertinib_evo_killer():
    cells_measurements = UlixertinibDataset.load_measurements()
    cells_drug_name = UlixertinibDataset.get_drug_name()

    cells_measurements_8h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 8").loc[:,cells_measurements.columns[1:-1]].to_numpy()) # type: ignore
    means_8h = cells_measurements_8h.mean(axis=0, keepdims=True)
    stds_8h = cells_measurements_8h.std(axis=0, keepdims=True)

    cells_measurements_24h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 24").loc[:,cells_measurements.columns[1:-1]].to_numpy()) # type: ignore
    means_24h = cells_measurements_24h.mean(axis=0, keepdims=True)
    stds_24h = cells_measurements_24h.std(axis=0, keepdims=True)

    cells_measurements_48h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 48").loc[:,cells_measurements.columns[1:-1]].to_numpy()) # type: ignore
    means_48h = cells_measurements_48h.mean(axis=0, keepdims=True)
    stds_48h = cells_measurements_48h.std(axis=0, keepdims=True)

    def checkpoint_8h_24h(t, x, std_threshold=2., cutoff=.2):
        effective_t = (t - cells_time(8)) / (cells_time(24) - cells_time(8))
        effective_mean = effective_t * means_24h + (1-effective_t) * means_8h
        effective_std = effective_t * stds_24h + (1-effective_t) * stds_8h
        threshold_violation_score = jnp.mean(jnp.abs(x - effective_mean) > std_threshold * effective_std, axis=1)
        return (threshold_violation_score < cutoff) * 0. + (threshold_violation_score >= cutoff) * threshold_violation_score

    def checkpoint_24h_48h(t, x, std_threshold=1.5, cutoff=.3):
        effective_t = (t - cells_time(24)) / (cells_time(48) - cells_time(24))
        effective_mean = effective_t * means_48h + (1-effective_t) * means_24h
        effective_std = effective_t * stds_48h + (1-effective_t) * stds_24h
        
        # Just consider the first 11 dimensions
        # relevant_dims = slice(None, None, None)
        relevant_dims = slice(12, None, None)

        threshold_violation_score = jnp.mean((jnp.abs(x - effective_mean) > std_threshold * effective_std)[...,relevant_dims], axis=1)

        return (threshold_violation_score < cutoff) * 0. + (threshold_violation_score >= cutoff) * threshold_violation_score * effective_t
    

    return UlixertinibDataset.build_killer(checkpoint_8h_24h, checkpoint_24h_48h)


def trametinib_erlotinib_cells_measurement_density():
    cells_measurements = TrametinibErlotinibDataset.load_measurements()
    cells_drug_name = TrametinibErlotinibDataset.get_drug_name()

    cells_measurements_8h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 8").loc[:,cells_measurements.columns[1:-1]].to_numpy()) # type: ignore
    means_8h = cells_measurements_8h.mean(axis=0, keepdims=True)
    stds_8h = cells_measurements_8h.std(axis=0, keepdims=True)

    cells_measurements_24h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 24").loc[:,cells_measurements.columns[1:-1]].to_numpy()) # type: ignore
    means_24h = cells_measurements_24h.mean(axis=0, keepdims=True)
    stds_24h = cells_measurements_24h.std(axis=0, keepdims=True)

    cells_measurements_48h = jnp.array(cells_measurements.loc[cells_measurements['Condition'] == cells_drug_name].query("time == 48").loc[:,cells_measurements.columns[1:-1]].to_numpy()) # type: ignore
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

    density_kernel = gaussian_kernel()

    def weighted_density_8h_24h(t, pos):
        return kde(density_kernel, cells_measurements_24h, pos)
    
    def weighted_density_24h_48h(t, pos):
        return kde(density_kernel, cells_measurements_48h, pos)

    def _killer(t, pos):
        mean_based_transitions = (cells_time(8) <= t) * (t < cells_time(24)) * jnp.clip(-1. + checkpoint_8h_24h(t, pos), None, 0.) * 1e0 + (cells_time(24) <= t) * checkpoint_24h_48h(t, pos, std_threshold=(2. - .5 * (t - cells_time(24)) / (cells_time(48) - cells_time(24)))) * 5e0
        density_based_transitions = (cells_time(8) <= t) * (t < cells_time(24)) * jnp.square(weighted_density_8h_24h(t, pos)) * (-1.) + (cells_time(24) <= t) * jnp.square(weighted_density_24h_48h(t, pos)) * (-1.)

        return mean_based_transitions + 10 * density_based_transitions
    
    return _killer


def format_date(date):
    return datetime.datetime.strptime(date, "%Y-%m-%d")

class CovidDataset(Dataset):

    @staticmethod
    def get_dataset_version():
        dataset_version = "v4"
        return dataset_version

    @staticmethod
    def get_dataset_name():
        dataset_version = CovidDataset.get_dataset_version()
        dataset_folder = "../data/covid/"
        dataset_name = os.path.join(dataset_folder, f"covid_{dataset_version}")

        return dataset_name
    
    @staticmethod
    def load_country_embs():
        dataset_name = CovidDataset.get_dataset_name()
        country_embs = pd.read_csv(f"{dataset_name}_country_embs.csv", index_col=0)

        return country_embs
    
    @staticmethod
    def load_variant_embs():
        dataset_name = CovidDataset.get_dataset_name()
        variant_embs = jnp.load(f"{dataset_name}_antigen_embs.npz")

        return variant_embs
    
    @staticmethod
    def load_ground_truth_evolution():
        dataset_name = CovidDataset.get_dataset_name()
        ground_truth_evolution = pd.read_csv(f"{dataset_name}_evolution_ground_truth.csv")
        ground_truth_evolution['date'] = pd.to_datetime(ground_truth_evolution['date'])
        ground_truth_evolution = ground_truth_evolution.set_index(["location", "date"])

        return ground_truth_evolution
    
    @staticmethod
    def load_sequence_nums(global_stats=True):
        dataset_name = CovidDataset.get_dataset_name()
        ground_truth_sequence_num = pd.read_csv(f"{dataset_name}_tot_sequences_ground_truth.csv")
        ground_truth_sequence_num["date"] = pd.to_datetime(ground_truth_sequence_num["date"])
        ground_truth_sequence_num = ground_truth_sequence_num.set_index(["location", "date"]).unstack(1).droplevel(0, axis=1)

        if global_stats:
            return ground_truth_sequence_num.sum(axis=0)
        else:
            return ground_truth_sequence_num
    
    init_date = format_date("2021-04-05")
    end_date = format_date("2021-08-09")
    
    @staticmethod
    def covid_time(date):
        return (format_date(date) - CovidDataset.init_date) / (CovidDataset.end_date - CovidDataset.init_date)

    def __init__(self, dataset_name, state_dims, killing_function_name) -> None:
        super().__init__(dataset_name, state_dims, killing_function_name)

        dataset_name = CovidDataset.get_dataset_name()

        self.t0_points = jnp.load(f"{dataset_name}_t0_points.npy")
        self.t1_points = jnp.load(f"{dataset_name}_t1_points.npy")

        # Load names of variant (in order, as they appear in t0 and t1)
        self.variant_names = pd.read_csv(f"{dataset_name}_variant_names.csv")['0']

        self.antigen_embs = CovidDataset.load_variant_embs()

        self.country_embs = CovidDataset.load_country_embs()
        self.np_country_embs = self.country_embs.apply(lambda x: x.to_numpy(), axis=1)

        self.std = self.t1_points.std(axis=0, keepdims=True)

    def close_to(self, pos, country):
        dist = jnp.sqrt(jnp.square(pos - self.np_country_embs.loc[country].reshape((1,-1))).sum(axis=-1))
        closeness = jnp.clip(1. - dist, 0.)
        return closeness

    def f(self, t, x):
        drift = jnp.zeros_like(x)
        drift = drift.at[...,2:].set(self.t1_points[...,2:].mean(axis=0, keepdims=True) - self.t0_points[...,2:].mean(axis=0, keepdims=True))

        # Delta drift
        time_constr = (t <= self.covid_time('2021-05-20'))
        place_constr = self.close_to(x[:,:2], "India")
        drift = drift.at[...,2:].set(drift[...,2:] + time_constr * place_constr.reshape((-1, 1)) * .1 * (self.antigen_embs['Delta'] - x[:,2:]))

        return drift

    def g(self, t, x):
        return jnp.array([[triangular_diffusivity(t, 2.0, .6)] * 2 +  [.15] * (x.shape[-1]-2)]) * self.std

    def project(self, state):
        return state[...,:2]

    def as_time(self, day):
        return day

    def pi_0_sample(self, key, n_samples=300):
        return self.t0_points.at[random.permutation(key, self.t0_points.shape[0])[:n_samples]].get()

    def pi_1_sample(self, key, n_samples=300):
        return self.t1_points.at[random.permutation(key, self.t1_points.shape[0])[:n_samples]].get()


def delta_evolution():

    country_embs = CovidDataset.load_country_embs().apply(lambda x: x.to_numpy(), axis=1)
    variant_embs = CovidDataset.load_variant_embs()

    margin = .1
    world_min_boundary = jnp.stack(country_embs.to_list()).min(axis=0).reshape((-1,2)) - margin
    world_max_boundary = jnp.stack(country_embs.to_list()).max(axis=0).reshape((-1,2)) + margin

    def close_to(pos, country):
        dist = jnp.sqrt(jnp.square(pos - country_embs.loc[country].reshape((1,-1))).sum(axis=-1))
        closeness = jnp.clip(2. - dist, 0.)
        return closeness
    
    country_embs_array = jnp.stack(country_embs)
    def far_from_countries(pos):
        country_dist = vmap(lambda x: jnp.sqrt(jnp.square(x.reshape((1,-1)) - country_embs_array).sum(axis=-1)).min(axis=0))(pos)
        country_dist = jnp.clip(country_dist, 0., .5)
        return country_dist
    
    def close_to_countries(pos):
        country_dist = vmap(lambda x: jnp.sqrt(jnp.square(x.reshape((1,-1)) - country_embs_array).sum(axis=-1)).min(axis=0))(pos)
        country_closeness = jnp.clip(.5 - country_dist, 0.)
        return country_closeness
    

    def variant_is(emb, variant):
        variant_dist = jnp.sqrt(jnp.square(emb - variant_embs[variant].reshape((1,-1))).sum(axis=-1))
        closeness = jnp.clip(.5 - variant_dist, 0.)
        return closeness

    as_time = CovidDataset.covid_time

    def _killer(t, x):
        pos = x[:,:2]
        embs = x[:,2:]

        india_delta_appearance = (t <= as_time("2021-05-20")) * close_to(pos, "India") * variant_is(embs, "Delta") * (-5e2)
        
        across_countries_disappearance = far_from_countries(pos)
        at_country_appearance = close_to_countries(pos)

        in_world_boundaries = jnp.logical_or((pos < world_min_boundary).any(axis=-1), (world_max_boundary < pos).any(axis=1)) * 20.

        return india_delta_appearance + in_world_boundaries + across_countries_disappearance * (5e-1) + at_country_appearance * (-1e2)
    
    return _killer


def covid_evolution():

    country_embs = CovidDataset.load_country_embs().apply(lambda x: x.to_numpy(), axis=1)
    variant_embs = CovidDataset.load_variant_embs()

    # TODO: Assumption. Files containing variant emebeddings are stored in **alphabetical** order
    variant_emb_matrix = jnp.expand_dims(jnp.concatenate([variant_embs[v] for v in variant_embs.files], axis=0), axis=0)  # shape: (1,var_num,d)

    ground_truth_evolution = CovidDataset.load_ground_truth_evolution()

    # common_countries = country_embs.join(ground_truth_evolution, how="inner").index.get_level_values(0).unique().to_numpy()
    worldwide_mix = ground_truth_evolution.groupby("date").agg('mean')
    
    as_time = CovidDataset.covid_time
    worldwide_mix_times = jnp.array(worldwide_mix.index.astype(str).to_series().apply(as_time).to_numpy())
    
    neg_mix_increment = jnp.array(-(worldwide_mix.to_numpy()[1:] - worldwide_mix.to_numpy()[:-1]))
    # Set minimum universal death rate of "others" variant
    neg_mix_increment = neg_mix_increment.at[:,-1].set(jnp.clip(neg_mix_increment[:,-1], 1e2))

    margin = .1
    world_min_boundary = jnp.stack(country_embs.to_list()).min(axis=0).reshape((-1,2)) - margin
    world_max_boundary = jnp.stack(country_embs.to_list()).max(axis=0).reshape((-1,2)) + margin

    def close_to(pos, country):
        dist = jnp.sqrt(jnp.square(pos - country_embs.loc[country].reshape((1,-1))).sum(axis=-1))
        closeness = jnp.clip(2. - dist, 0.)
        return closeness
    
    def variant_is(emb, variant):
        variant_dist = jnp.sqrt(jnp.square(emb - variant_embs[variant].reshape((1,-1))).sum(axis=-1))
        closeness = jnp.clip(.5 - variant_dist, 0.)
        return closeness


    def _killer(t, x):
        pos = x[:,:2]
        embs = x[:,2:]

        variant_prevance_neg_increment = (jnp.expand_dims((worldwide_mix_times[:-1] <= t) * (t < worldwide_mix_times[1:]), axis=1) * neg_mix_increment).sum(axis=0, keepdims=True)  # shape: (1,var_num)

        exp_embs = jnp.expand_dims(embs, axis=1)  # shape: (n,1,d)
        closeness_to_variants = 1/(2+jnp.square(exp_embs - variant_emb_matrix).sum(axis=-1))  # shape: (n,var_num)

        # Deaths: variants close to those that disappear (>> 0 neg. increment)
        variant_death_contributions = (variant_prevance_neg_increment > 0.) * (variant_prevance_neg_increment * closeness_to_variants)  # shape: (n,var_num)
        death_rate = variant_death_contributions.sum(axis=-1)  # shape: (n,)

        # Births: variants close to those that appear (<< 0 neg. increment)
        variant_birth_contributions = (variant_prevance_neg_increment < 0.) * (variant_prevance_neg_increment * closeness_to_variants)  # shape: (n,var_num)
        birth_rate = variant_birth_contributions.sum(axis=-1)  # shape: (n,)

        in_world_boundaries = jnp.logical_or((pos < world_min_boundary).any(axis=-1), (world_max_boundary < pos).any(axis=1))

        india_delta_appearance = (t < as_time("2021-05-25")) * close_to(pos, "India") * variant_is(embs, "Delta")

        return (1e-1) * death_rate + (5) * birth_rate + in_world_boundaries + (-1e1) * india_delta_appearance
    
    return _killer


class CiteseqDataset(Dataset):

    @staticmethod
    def get_dataset_name():
        dataset_folder = "../data/neurips22/"

        version = "v3"
        donor = 31800
        cell_type = "hsc"

        dataset_name = os.path.join(dataset_folder, f"citeseq-{version}_{donor}_{cell_type}")

        return dataset_name
    
    @staticmethod
    def load_measurements():
        dataset_version_name = CiteseqDataset.get_dataset_name()

        measurements = {
            day: jnp.load(f"{dataset_version_name}_day-{day}.npy")
        for day in [2, 3, 4]}

        return measurements
    
    @staticmethod
    def citeseq_time(day):
        return (day - 2.) / (4. - 2.)

    def __init__(self, dataset_name, state_dims, killing_function_name) -> None:
        super().__init__(dataset_name, state_dims, killing_function_name)

        dataset_name = CiteseqDataset.get_dataset_name()

        self.measurements = CiteseqDataset.load_measurements()
        self.projection = batch_project(joblib.load(f"{dataset_name}_pca-2.pkl"))

        self.means = {day: self.measurements[day].mean(axis=0) for day in self.measurements}
        self.std = {day: self.measurements[day].std(axis=0) for day in self.measurements}

    def f(self, t, x):
        # return self.mean
        return (self.as_time(2) <= t) * (t < self.as_time(3)) * (self.means[3] - self.means[2]) + (self.as_time(3) <= t) * (self.means[4] - self.means[3])
        # return 0.
    
    def g(self, t, x):
        return triangular_diffusivity(t, .5, .15) * ((self.as_time(2) <= t) * (t < self.as_time(3)) * self.std[3] + (self.as_time(3) <= t) * self.std[4])
        # return 2. * self.std

    def project(self, state):
        return self.projection(state)

    def as_time(self, day):
        return CiteseqDataset.citeseq_time(day)

    def pi_0_sample(self, key, n_samples=300):
        return self.measurements[2].at[random.permutation(key, self.measurements[2].shape[0])[:n_samples]].get()

    def pi_1_sample(self, key, n_samples=300):
        return self.measurements[4].at[random.permutation(key, self.measurements[4].shape[0])[:n_samples]].get()


def citeseq_measurement_killer():

    measurements = CiteseqDataset.load_measurements()

    as_time = CiteseqDataset.citeseq_time

    cell_means = {day: measurements[day].mean(axis=0) for day in measurements}
    cell_stds = {day: measurements[day].std(axis=0) for day in measurements}

    def checkpoint(t, x, start_day, end_day, std_threshold=2., cutoff=.2):
        effective_t = (t - as_time(start_day)) / (as_time(end_day) - as_time(start_day))
        effective_mean = effective_t * cell_means[end_day] + (1-effective_t) * cell_means[start_day]
        effective_std = effective_t * cell_stds[end_day] + (1-effective_t) * cell_stds[start_day]
        threshold_violation_score = jnp.mean(jnp.abs(x - effective_mean) > std_threshold * effective_std, axis=1)
        return (threshold_violation_score < cutoff) * 0. + (threshold_violation_score >= cutoff) * threshold_violation_score * effective_t

    def _citeseq_measurement_killer(t, pos):
        return (as_time(2) <= t) * (t < as_time(3)) * checkpoint(t, pos, 2, 3, cutoff=.001) * 1e-1 + (as_time(3) <= t) * (1/(1+checkpoint(t, pos, 3, 4, cutoff=0.01))) * (-1e-3)
    
    return _citeseq_measurement_killer



## Killing functions

def no_killing(t, pos):
    return jnp.zeros(pos.shape[:-1])

def centered_rectangle_dead_pool(t, pos):
    dead_zone_mask = (2 < pos[:,0]) * (pos[:,0] < 3) * (-5 <= pos[:,1]) * (pos[:,1] < 5)

    return dead_zone_mask * .5

def rectangle_death_birth(t, pos):
    birth_zone_mask = (2 < pos[:,0]) * (pos[:,0] < 3) * (-5 <= pos[:,1]) * (pos[:,1] < 5)
    dead_zone_mask = (-3 < pos[:,0]) * (pos[:,0] < -2) * (-5 <= pos[:,1]) * (pos[:,1] < 5)

    return dead_zone_mask * .5 - birth_zone_mask * .5

def double_strength_death_rectangle(t, pos):
    weak_dead_zone_mask = (2 < pos[:,0]) * (pos[:,0] < 3) * ((-10 <= pos[:,1]) * (pos[:,1] < -7) + (7 <= pos[:,1]) * (pos[:,1] < 10))
    strong_dead_zone_mask = (2 < pos[:,0]) * (pos[:,0] < 3) * (-7 <= pos[:,1]) * (pos[:,1] < 7)

    return weak_dead_zone_mask * 1e-4 + strong_dead_zone_mask * jnp.inf

def split_rectangle_dead_pool(t, pos):
    lower_death_zone = (0 < pos[:,0]) * (pos[:,0] < 1) * (-10 <= pos[:,1]) * (pos[:,1] < -1)
    higher_death_zone = (0 < pos[:,0]) * (pos[:,0] < 1) * (1 <= pos[:,1]) * (pos[:,1] < 10)

    return lower_death_zone * 1e1 + higher_death_zone * 1e1



# Available datasets
datasets = {
    'toy'  :            ToyDataset,
    'no-outlier':       OutlierCompleteDataset,
    'outlier-weak':     OutlierWeakDataset,
    'outlier':          OutlierDataset,
    'outlier-birth':    OutlierBirthDataset,
    'categorical':      CategoricalDataset,
    'covid':            CovidDataset,
    'cells':            TrametinibErlotinibDataset,
    'cells-ulixertinib':UlixertinibDataset,
    'citeseq':          CiteseqDataset,
}

# Available killing functions
killing_funcs = {
    'no_killing':                                       no_killing,

    # Outlier (death)
    'outlier_death':                                    outlier_death,
    'outlier_death_birth':                              outlier_death_birth,
    'progressive_death':                                progressive_death,

    # Toy
    'centered_rectangle_dead_pool':                     centered_rectangle_dead_pool,
    'rectangle_death_birth':                            rectangle_death_birth,
    'double_strength_death_rectangle':                  double_strength_death_rectangle,
    'split_rectangle_dead_pool':                        split_rectangle_dead_pool,
    'categorical_split_rectangle':                      categorical_split_rectangle,
    
    # Covid
    'delta_evolution':                                  delta_evolution(),
    'covid_evolution':                                  covid_evolution(),

    # Cells
    # 'ulixertinib_cells_measurement_killer':             cells_measurement_killer(UlixertinibDataset),
    'trametinib_erlotinib_cells_measurement_killer':    cells_measurement_killer(TrametinibErlotinibDataset),
    'trametinib_erlotinib_cells_measurement_density':   trametinib_erlotinib_cells_measurement_density(),
    # 'ulixertinib_evo_killer':                           ulixertinib_evo_killer(),

    # Citeseq
    # 'citeseq_measurement_killer':                       citeseq_measurement_killer(),
}