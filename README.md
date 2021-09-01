# Matrix Models with Netket

Find the ground state of matrix models using variational neural states in [`netket`](https://www.netket.org).

This repository depends on `netket`.

We included a tutorial based on the nice `Introduction to Netket 3.0` tutorial at [this link](https://www.netket.org/tutorials/netket3.html).

# Install

The documentation page for [`netket`](https://www.netket.org/getting_started.html) suggests to create a python virtual environment and then use `pip` to install the library.
In this repository we work with the serial CPU version of `netket` and we create a minimal virtual environment using [`conda`](https://docs.conda.io/projects/conda/en/latest/) based on the [environment.yml](./environment.yml) file:

```bash
conda env create -f environment.yml
conda activate netket
```

# Notebooks

The [MatrixModel.ipynb](./notebooks/MatrixModel.ipynb) notebooks is an introduction to using `netket` for finding the variational ground state of a bosonic matrix model using a neural state ansatz and a variational Monte Carlo sampler.