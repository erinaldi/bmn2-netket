# Matrix Models with Netket

Find the ground state of matrix models using variational neural states in [`netket`](https://www.netket.org).

This repository depends on the `netket` library (v3.0).

We included a notebook based on the nice `Introduction to Netket 3.0` tutorial at [this link](https://www.netket.org/tutorials/netket3.html).

# Install

The documentation page for [`netket`](https://www.netket.org/getting_started.html) suggests to create a python virtual environment and then use `pip` to install the library.
In this repository we work with the serial CPU version of `netket` and we create a minimal virtual environment using [`conda`](https://docs.conda.io/projects/conda/en/latest/) based on the [environment.yml](./environment.yml) file:

```bash
conda env create -f environment.yml
conda activate netket
```

**Note**: `netket` is isntalled in the environment using `pip`. The installation is for a single CPU (no MPI installation is needed).
# Notebooks

The [MatrixModel.ipynb](./notebooks/MatrixModel.ipynb) notebooks is an introduction to using `netket` for finding the variational ground state of a bosonic matrix model using a neural state ansatz and a variational Monte Carlo sampler for states in a Fock basis.
The simpler case of zero coupling constant is shown in [HarmonicOscillators.ipynb](./notebooks/HarmonicOscillators.ipynb).

You can run the notebooks on Google Colaboratory: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erinaldi/bmn2-netket/blob/main/notebooks/MatrixModel.ipynb)