# A Bayesian-Symbolic Approach to Reasoning and Learning in Intuitive Physics

This GitHub [repository](https://github.com/xukai92/bsp/) contains code and data for our paper *A Bayesian-Symbolic Approach to Reasoning and Learning in Intuitive Physics* at NeurIPS 2021.
Please cite our paper using the following BibTeX entry if you use this source code in your research.
```
@inproceedings{xu2021bsp,
  title={A {B}ayesian-Symbolic Approach to Reasoning and Learning for Intuitive Physics},
  author={Kai Xu and Srivastava, Akash and Gutfreund, Dan and Sosa, Felix and Ullman Tomer and Tenenbaumm, Joshua B. and Sutton, Charles},
  booktitle={The 35th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021},
}
```

`/`
- `data/`: generated data and processed data
    - `phys101`: selected scenes from Physics 101 (Wu et al., 2016); originally from http://phys101.csail.mit.edu/
    - `synth`: 3 types of synthetic scenes
    - `ullman`: visual stimulus used in Ullman et al. (2019); originally from https://github.com/tomeru/LPDS/
- `notebooks/`: interactive notebooks to extract or analyse experimental results
    - `demo-world.ipynb`: example notebook to demonstrate how to run simulation with `BayesianSymbolic.jl`
    - `monly.ipynb`: notebook to extract results for M-step only results
    - `em.ipynb`: notebook to extract results for complete EM results
    - `phys101.ipynb`: notebook to extract results for PHYS101 results
    - `ullman.ipynb`: notebook to extract results for ULLMAN results
- `paper/`: interactive notebooks to produce artifects for the paper
    - `raw/`: extracted results
    - `figures.ipynb`: notebook to make figures
    - `tables.ipynb`: notebook to make tables
- `scripts/`: executable scripts to generate or pre-process data or to run experiments
    - `evalexpr_efficiency.jl`: script to compare different implementations for expression evaluations
        - This does not affect the results but only performance.
    - `generate-synth.jl`: script to generate SYNTH
    - `helper.jl`: common helper functions used by scripts
    - `neural.jl`: concrete architecture and hyper-parameters for neural baselines
    - `preprocess-ullman-master.jl`: master script to process a scene from ULLMAN, calling `scripts/preprocess-ullman.py`
    - `preprocess-ullman.py`: script to process scenes from ULLMAN
    - `runexp-ullman.jl`: script to run an experiment on ULLMAN; for other datasets, use `runexp.jl`
    - `runexp.jl`: script to run an experiment on a given dataset and specific hyper-parameters
        - All datasets but ULLMAN are supported; for ULLMAN, use `runexp-ullman.jl`
    - `ullman_hacks.jl`: ULLMAN specific grammar 
        - See the comments on hacks in the beginning of the file
- `src/`: source codes for simulation and the BSP algorithm
    - `BayesianSymbolic.jl/`: world construction and simulations
    - `data/`: data processing functions
        - `phys101.jl`: PHYS101 specific functions
        - `preprocessing.jl`: generic data preparation functions
        - `ullman.jl`: ULLMAN specific functions
        - `ullman.txt`: ground truth information for ULLMAN
    - `scenarios/`: scenarios implemented in Turing.jl
        - `bounce.jl`: the BOUNCE scenario from SYNTH
        - `fall.jl`: the FALL scenario from PHYS101
        - `magnet.jl`: the MAGNET scenario
            - This is not used in the paper
        - `mat.jl`: the MAT scenario from SYNTH
        - `nbody.jl`: the NBDOY scenario from SYNTH
        - `spring.jl` the SPRING scenario from PHYS101
        - `ullman.jl`: the scenario from ULLMAN
    - `analyse.jl`: quantitative and visual analysis
    - `app_inf.jl`: functions for approximate inference
    - `dataset.jl`: functions for loading datasets (SYNTH, PHYS101 and ULLMAN)
    - `exp_max.jl`: types and interfaces for the EM algorithm
    - `neural.jl`: generic implementations of neural baselines
    - `sym_reg.jl`: functions for symbolic regression
    - `utility.jl` utility functions for simulation and loss computation
- [`suppl/`](https://github.com/xukai92/bsp/tree/main/suppl/): supplementary materials
    - `bounce_inspection`: visualisation for results discussed in appendix C.3.1
    - `generalization`: visualisation for results discussed in appendix C.3.2
- `Manifest.toml`: the exact package version of this environment
- `master.jl`: master scripts to run a batch of experiments, calling scripts in `scripts/`
- `Project.toml`: the dependency of this environment
- `README.md`: this file you are reading

## Setups

BSP is implemented with Julia and some of the dependencies or scripts also rely on Python.

### Julia

Please follow https://julialang.org/downloads/ to download and install Julia.
Make sure `julia` is available in your executable path.
Then from *the root of this repo*, you can do `julia -e "import Pkg; Pkg.instantiate()"` to instantiate the environment.

There are a few more steps to have Julia properly linked with Python, which is explained next.

### Python

You will need to have Python installed and a virtual environment set up.
The virtual environment should have the following packages
- `matplotlib`
- `pandas`
- `wandb`
To properly link this virtual environment with Julia, please follow https://github.com/JuliaPy/PyCall.jl.

You will also have all necessary Python dependencies to run `scripts/preprocess-ullman.py`.
Please see the libraries imported in the script.

Once these steps are done, you are ready to run the scripts.

## Reproducing results

### Figure 4

Run the following experiments
- `julia master.jl efficiency synth/nbody`
- `julia master.jl efficiency synth/bounce`
- `julia master.jl efficiency synth/mat`

Collect the results using `notebooks/monly.ipynb` and make the figure using `paper/figures.ipynb`

### Figure 5

Run the following experiments
- `julia master.jl ablation synth/nbody`
- `julia master.jl ablation synth/bounce`
- `julia master.jl ablation synth/mat`

Collect the results using `notebooks/monly.ipynb` and make the figure using `paper/figures.ipynb`

### Table 1

Run the following experiments
- `julia master.jl em synth/nbody`

Collect the results using `notebooks/em.ipynb` and make the figure using `paper/tables.ipynb`

### Table 2

Run the following experiments
- `julia master.jl phys101 fall`
- `julia master.jl phys101 spring`

Collect the results using `notebooks/phys101.ipynb` and make the figure using `paper/tables.ipynb`

### Table 3 & Figure 10

Run the following experiments
- `julia master.jl ullman`

Collect the results using `notebooks/ullman.ipynb` and make the table using `paper/tables.ipynb` & figure using `paper/figures.ipynb`

## Misc

Set the environment variable `JULIA_NUM_THREADS=10` before running any scripts will enable multiple-threading (e.g. 10 threads in this example) whenever it's programmed to do so.
For example, `master.jl` executes a batch of experiments and it is programmed to run them in a multi-threading manner which can benefit from setting this environment variable.
