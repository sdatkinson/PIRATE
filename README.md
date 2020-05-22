# PIRATE
Physics-informed research assistant for theory extraction.
Code accompanying [Atkinson et al., "Data-driven discovery of free-form governing differential equations" (2019)](https://arxiv.org/abs/1910.05117)

## Getting set up
```bash
conda env create -f environment.yml
conda activate pirate
pip install .
```

## Running the experiments:
To re-create Figure 3a, Run the four scripts in `examples/non_adaptive`, varying 
the `n_train` and `seed` flags.

To re-create Figure 3b., run the scripts in `experiments/adaptive`.
