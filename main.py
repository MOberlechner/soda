from scripts.run_experiments import *

"""
General Parameter for Experiment

path_config     specify path to config file, which contain files for 
                learn_alg and experiment in a subdirectory setting

path            log files and strategies are saved in the specified directory


Experiment sepcific Parameter (which config file to access)
setting         specifies the mechanism / subdirectory in config
learn_alg       specifies learning algorithm (config file in learner subdirectory)
exp_lists       list of settings where we want to compute the equilibria
simulation      run simulations, otherwise only the strategies are computed

"""
path_config = "configs/"
path = "experiment/"

learn_alg = "frank_wolfe"
setting = "single_item"

exp_list = ["fpsb", "spsb"]

simulation = True

"""
Run experiment - Learn distributional strategies

logging (bool)      log information to runs
num_runs (int)      number of repetitions of each experiment
save_strat (bool)   save solutions
"""
logging = True
num_runs = 5
save_strat = True

for experiment in exp_list:
    run_experiment(
        learn_alg,
        setting,
        experiment,
        logging,
        save_strat,
        num_runs,
        path,
        path_config,
    )


"""
Run simulations - evaluate computed strategies

n_obs (int)     number of simulated auctions used to compare with analytical BNE
n_scaled (int)  if no analytical BNE is available we evaluate the strategy in a discretized game with a higher discretization (slow)

"""

if simulation:
    n_obs = int(2**22)
    n_scaled = m_scaled = 1024

    for experiment in exp_list:
        run_sim(
            learn_alg,
            setting,
            experiment,
            path,
            path_config,
            num_runs,
            n_obs,
            logging,
            n_scaled,
            m_scaled,
        )
