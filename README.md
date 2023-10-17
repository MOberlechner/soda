# SODA

### Structure of the Code
There are four central classes
- [Mechanism](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/src/mechanism/mechanism.py) - represents the auction/contest mechanism we want to consider
- [Game](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/src/game.py) - represents the discretized version of the mechanism
- [Strategy](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/src/strategy.py) - represents the distributional strategy we want to compute
- [Learner](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/src/learner/learner.py) - represents the learning algorithm we use to compute the equilibrium strategy

The basic functionality can be seen in the [intro notebook](https://gitlab.lrz.de/ga38fip/soda/-/blob/main/notebooks/intro.ipynb).

---

### Notation
Some notes on the general notation used in this project

| Variable | Description |
| --- | --- |
|`mechanism` | Describes the underlying mechanis,<br> e.g., all-pay auction, single-item auction, contest, ... |
| `setting` |  A setting is a specific instance of a mechanism, <br> e.g. FPSB (single-item auction), with uniform prior and two agents. |
| `game` | The discretized version of a specific setting |
| `learner` | Denotes a specific learning method, <br> e.g., SODA, SOMA, ... |
| `experiment` | An experiment consists of a setting (-> game) + learner. |
| `experiment_tag` | Allows us to group experiments by using the same tag, <br> e.g., risk-aversion might contain different settings with different levels of risk aversion
---


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
