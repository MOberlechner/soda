# Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging (SODA)
This code is provided for academic research purposes only. This code is not licensed for commercial use.
If you find this code helpful and use it in your research, please cite the following paper:

>**Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging.**<br>
*Martin Bichler, Maximilian Fichtl, Matthias Oberlechner*<br>
Operations Research, 2023 (Forthcoming)

## Projects
This repository contains numerical experiments for the following publications:

1. Project soda:

2. Project contests

## What is implemented?

We focus on incomplete-information (Bayesian) games with continuous type and action space. 
By discretizing the type and action space, and using distributional strategies, we can apply standard (gradient-based) learning algorithms to approximate Bayes-Nash equilibria (BNE) of given mechanisms.

#### Mechanisms

- Single-Item Auctions (first- and second-price auctions with risk-aversion, ...)
- All-Pay Auctions (first- and second-price, i.e. War of attrition)
- LLG-Auction (small combinatorial auction with 2 items, 2 local bidders, and 1 global bidder)
- Split-Award Auction (procurement auction with 2 agents)
- Tullock Contests

#### Learning Algorithms

- Dual Averaging (Gradient ascent with lazy projection, exponentiated gradient ascent)
- Mirror Ascent (projected gradient ascent, exponentiated gradient ascent)
- Frank-Wolfe Algorithm
- Fictitious Play


## Setup

Note: These setup instructions assume a Linux-based OS and uses python 3.8.10 (or higher).

Install virtualenv (or whatever you prefer for virtual envs)
```bash
sudo apt-get install virtualenv
```

Create a virtual environment with virtual env (you can also choose your own name)

```bash
virtualenv venv
```

You can specify the python version for the virtual environment via the -p flag. 
Note that this version already needs to be installed on the system (e.g. `virtualenv - p python3 venv` uses the 
standard python3 version from the system).

activate the environment with
```bash
source ./venv/bin/activate
```
Install all requirements

```bash
pip install -r requirements.txt`
```
Install the soda package.

```bash
pip install -e .
```
You can also run "pip install ." if you don't want to edit the code. The "-e" flag ensures that pip does not copy the code but uses the editable files instead.

## Install pre-commit hooks (for development)
Install pre-commit hooks for your project

```bash
pre-commit install
```

Verify by running on all files:

```bash
pre-commit run --all-files
```

For more information see https://pre-commit.com/.