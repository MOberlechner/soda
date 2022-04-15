# SODA
New and better implementation of SODA ...


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
