# Project: Contests
These are the numerical experiments for:

>**Computing Bayes Nash Equilibrium Strategies in Crowdsourcing Contests**<br>
*Martin Bichler, Markus Ewert, Matthias Oberlechner.*<br>
In 32nd Workshop on Information Technologies and Systems (WITS-22), Copenhagen, Denmark.


---

### Simulation
- the results will be stored in<br> `PATH_TO_EXPERIMENTS = "experiments/contests/"` 
- the config files can be found in<br> `PATH_TO_CONFIGS = "projects/contests/configs/"`

To compute the strategies (learning) and evaluate them (simulation) run the following scripts (from the main directory of the repo).

**Exp 1 - Crowdsourcing Contest**<br>
*Number of settings: 9 (10 runs each), Runtime: 13 min*  <br>
Experiments to compute BNE for different crowdsourcing contests with 3, 5, and 10 agents and 2 prices
```bash
python projects/soda/simulation/run_crowdsourcing.py
```

**Exp 2 - Generalized Tullock Contest**<br>
*Number of settings: 8 (10 runs each), Runtime: 8 min*  <br>
Experiments to compute BNE for Tullock contests with different impact parameter.
```bash
python projects/soda/simulation/run_tullock.py
```
Note that we changed the discretization to 64 x 64 for all settings.

### Evaluation
To create the plots and tables (for SODA) run the following scripts
```bash
python projects/soda/evaluation/create_table.py
python projects/soda/evaluation/create_plots.py
```