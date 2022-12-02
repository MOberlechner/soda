from src.util.run import *


def main(
    path_config: str,
    path: str,
    experiments: list,
    learning: bool,
    simulation: bool,
    logging: bool,
    n_obs: int = int(2**22),
    n_scaled: int = 1024,
    m_scaled: int = 1024,
):
    """Run Experiments

    Args:
        path_config (str): specify path to config file, which contain files for learn_alg and experiment in a subdirectory setting
        path (str): log files and strategies are saved in the specified directory
        experiments (list): list of tuples (setting, experiment, learner)
        setting (str): specifies the mechanism / subdirectory in config
        exp_list (list): list of specific version of the setting where we want to compute the equilibria
        learning (bool): run learning algorithm, otherwise only simulations are performed
        simulation (bool): run simulations, otherwise only the strategies are computed
        logging (bool): log results from simulation
        n_obs (int): number of simulated observations. Defaults to int(2**22).
        n_scaled (int): number of discretization points for observations to evaluate computed strategy. Defaults to 1024.
        m_scaled (int) number of discretization points for actions to evaluate computed strategy. Defaults to 1024.
    """

    # compute strategies
    if learning:
        for setting, experiment, learn_alg in experiments:
            print(setting, experiment, learn_alg)
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

    # evaluate strategies
    if simulation:
        for setting, experiment, learn_alg in experiments:
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

    print("Done!")


if __name__ == "__main__":

    path_config = "configs/"
    path = "experiment/test/"
    experiments = [("single_item", "fpsb", "soda")]

    # computation
    learning = False
    num_runs = 2
    save_strat = True

    # simulation
    simulation = True
    logging = True

    main(path_config, path, experiments, learning, simulation, logging)
