# wqu-final-project

Code for our final project of the M.Sc. in Financial Engineering at WorldQuant University.

## Authors

Ryan Cardenas, Qihu Zhang, and Rakotoarimanana Nomenjanahary (Hosea Levi)

# Installation

To install dependencies after cloning this repo, open a terminal and navigate to this repo on your machine. Then run `bash setup.sh`, which will perform the following operations:

1. Install Miniconda if you don't already have Conda installed.
1. Creates the `wqufp` Conda environment from `environment.yml` if `wqufp` does not already exist.
1. Activates the `wqufp` Conda environment.
1. Installs pre-commit hooks required for contributing to this project.

After installing, any time you want to run the code in this project, you can simply run `conda activate wqufp` to load the environment dependencies.
