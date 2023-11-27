# Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging (SODA)
This code is provided for academic research purposes only. This code is not licensed for commercial use.
If you find this code helpful and use this code in your research, please cite the following paper:

>**Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging.**<br>
*Martin Bichler, Maximilian Fichtl, Matthias Oberlechner*<br>
Operations Research, 2023 (Forthcoming)

## Projects
The code contains different projects and can be used to reproduce the respective results.

- **Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging.**<br>
*Martin Bichler, Maximilian Fichtl, Matthias Oberlechner*<br>
Operations Research, 2023 (Forthcoming) <br> See `projects/soda/readme` for experiments.
- (tbd)

## What is implemented?

We focus on incomplete-information (Bayesian) games with continuous type and action space. 
By discretizing the type and action space, and using distributional strategies, we can apply standard (gradient-based) learning algorithms to approximate Bayes-Nash equilibria (BNE) of given mechanisms.

#### Mechanisms

- Single-Item Auctions <br>*first- and second-price auctions with risk-aversion, different utility functions (quasi-linear, return-on-invest, return-on-spent)*
- All-Pay Auctions<br>
 *first- and second-price (war of attrition), risk aversion*
- LLG-Auction <br>
  *small combinatorial auction with 2 items, 2 local bidders, and 1 global bidder and correlated valuations of local bidders*
- Split-Award Auction <br> 
 *procurement auction with 2 agents and (dis-)economies of scale*
- Tullock Contests <br> *different discrimination parameters*

#### Learning Algorithms

- Dual Averaging <br> *Gradient ascent with lazy projection, exponentiated gradient ascent*
- Mirror Ascent <br> *projected gradient ascent, exponentiated gradient ascent*
- Frank-Wolfe Algorithm
- Fictitious Play

## Intro
A short explanation of the code can be found in the jupyter notebooks.

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
