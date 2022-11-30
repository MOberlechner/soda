# SODA

### Structure of the Code
There are four central classes
- [Mechanism](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/src/mechanism/mechanism.py) - represents the auction/contest mechanism we want to consider
- [Game](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/src/game.py) - represents the discretized version of the mechanism
- [Strategy](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/src/strategy.py) - represents the distributional strategy we want to compute
- [Learner](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/src/learner/learner.py) - represents the learning algorithm we use to compute the equilibrium strategy

The basic functionality can be seen in the [intro notebook](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/notebooks/intro.ipynb).

---

### References:
- **package opt_einsum:** Daniel G. A. Smith and Johnnie Gray, opt_einsum - A Python package for optimizing contraction order for einsum-like expressions. Journal of Open Source Software, 2018, 3(26), 753

- **projection onto simplex** is based on "W. Wang and M. A. Carreira-Perpinan - Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application" (https://arxiv.org/abs/1309.1541)


## Setup

Note: These setup instructions assume a Linux-based OS and uses python 3.8.10 (or higher).

Install virtualenv (or whatever you prefer for virtual envs)

`sudo apt-get install virtualenv`

Create a virtual environment with virtual env (you can also choose your own name)

`virtualenv venv`

You can specify the python version for the virtual environment via the -p flag. 
Note that this version already needs to be installed on the system (e.g. `virtualenv - p python3 venv` uses the 
standard python3 version from the system).

activate the environment with

`source ./venv/bin/activate`

Install all requirements

`pip install -r requirements.txt`

## Install pre-commit hooks (for development)
Install pre-commit hooks for your project

`pre-commit install`

Verify by running on all files:

`pre-commit run --all-files`

For more information see https://pre-commit.com/.
