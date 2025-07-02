To install the package, clone the repository and run the following commands:
```

conda create -n hpsenv python=3.12 petsc petsc4py -c defaults -c conda-forge

pip install torch
pip install jax[cuda12] # for GPU-capable machines, install the cuda version of jax
pip install -e .
git submodule update --recursive

```
