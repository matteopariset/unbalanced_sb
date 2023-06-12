# Unbalanced Diffusion Schrödinger Bridges

Reference repository for the paper **Unbalanced Diffusion Schrödinger Bridge**, cite as:

[put citation]

## Setup

### Environment setup

Create and activate a dedicated `conda` environment:

    conda env create -n udsb -f udsb.yml
    conda activate udsb

### Jax installation

Install `jax`:

    conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia

If you do **not** have hardware acceleration, you can find more information here: <https://github.com/google/jax#installation>.

Install remaning dependencies:

    pip install --upgrade dm-haiku==0.0.9

    conda install -c conda-forge optax
    conda install -c conda-forge ott-jax

### Dataset download

...


## Structure

Below, we detail the organization of this repository.

### Datasets

Raw data and preprocessing pipelines are contained in the `data` folder.
* `data/`:
   - `create_datasets.ipynb`: generate toy datasets
   - `prepare_cells.ipynb`: process **cell** drug response dataset
   - `prepare_flights.ipynb`: generate **country embedding** based on flights
   - `prepare_covid_variants.ipynb`: process **covid** 19 variant spread dataset
   - `2d/`: folder containing 2d representations of empirical distributions and killing zones
   - `4i/`: folder with raw and processed data belonging to the **cell** experiment
   - `flights/`: folder with flight data and country embedding
   - `covid/`: folder with raw and processed data belonging to the **covid** experiment


### Algorithms

The two algorithms presented in our paper are contained in:

* `udsb_td/`: code and experiments involving our UDSB-TD algorithm.
* `udsb_f/`: code and experiments involving our UDSB-F algorithm.

Both algorithms can be used with a similar interface. To _initialize_ an experiment, use:

    Experiment.create(config, ...)

and to reload a trained model with name `tag`, call:

    Experiment.load(dataset_name, tag)

Path sampling and plots can be obtained using the

    Viewer(key, experiment)

object, while training is perfomed via the snippet:

    trainer = Trainer(key, experiment)
    trainer.train(...)

For additional information, please refer to the documentation of **UDSB-F**.
We highlight here some invariants respected throughout our codebase:

 - Direction-dependent entities:
    1. Many entities exist in pair: one instance per SDE direction. When this happens, the pair is regrouped in a dictionary, indexed by the direction (`FORWARD` or `BACKWARD`).
    2. In the context of **training**, the direction assigned is the one for which the network is updated: e.g. _forward_ corresponds to the IPF pass in which the _forward_ score is learned.
    3. The names of direction-indexed dictionaries are singular. Examples: `model`, `ipf_loss`, `optimizer`, ....
    4. `broadcast()` provides a shortcut to execute a computation that is indexed by the direction.

 - Random operations:
    1. Functions that use randomness take a `key` argument.
    1. If a function returns a `key`, no splitting needs to be done by the caller: the function internally splits the key and returns a fresh state (e.g. for `key, ... = train(key, ...)`). Otherwise, the caller is in change of managing the state of the PRNG (as with `SDE.sample_f_trajectory()`).

 - We offer the following jit-compiled functions:
    1. `training_step`: which is used by `train()` to execute one IPF iteration.
    2. `fast_sample_trajectory_evo`: which is the faster alternative to the standard (slower) `sample_trajectory()` call (note that all functions in `Viewer` use the latter).


### Miscellaneous

The `reproducibility/` folder contains some pre-computed baseline predictions.
