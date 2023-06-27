# Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging (SODA)
The code was created for the following paper [[arxiv]](https://arxiv.org/abs/2208.02036):
```
@misc{bichler2023soda,
      title={Computing Bayes Nash Equilibrium Strategies in Auction Games 
      via Simultaneous Online Dual Averaging}, 
      author={Martin Bichler and Maximilian Fichtl and Matthias Oberlechner},  
      year={2023},  
      eprint={2208.02036}, 
      archivePrefix={arXiv},  
      primaryClass={cs.GT}  
}
```

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

**For development**: Install pre-commit hooks

`pre-commit install`

Verify by running on all files:

`pre-commit run --all-files`

For more information see https://pre-commit.com/.
