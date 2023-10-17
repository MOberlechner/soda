from typing import List

from soda.util.experiment import Experiment


def main(
    path_exp: str,
    experiment_list: List[tuple],
    number_runs: int,
    learning: bool,
    simulation: bool,
    logging: bool,
    number_samples: int,
    save_strat: bool,
    save_init_strat: bool = True,
    round_decimal: int = 3,
):
    """Run Experiments fom experiment_list, which are specified in respective config files

    Args:
        path_exp (str): path to directory where results are stored
        experiment_list (List[tuple]): list of experiments we want to perform, contains tuples (mechanism_type, experiment, learn_alg)
        number_runs (int): number of repetitions for each experiment
        learning (bool): learn strategies
        simulation (bool): simulate mechanisms with saved strategies to evaluate
        logging (bool): log results from computation/simulation
        number_samples (int, optional): number of samples for simulation.
        save_strat (bool): save strategies from computation
        save_init_strat (bool, optional): save initial strategies . Defaults to True.
        round_decimal (int, optional): Number of decimals we use for our metrics. Defaults to 3.
    """
    print(f"Running {len(experiment_list)} Experiments...\n")

    for config_game, config_learner in experiment_list:
        exp_handler = Experiment(
            config_game,
            config_learner,
            number_runs,
            learning,
            simulation,
            logging,
            save_strat,
            number_samples,
            save_init_strat,
            path_exp,
            round_decimal,
        )
        exp_handler.run()


if __name__ == "__main__":

    path_exp = "experiment/test/"

    experiment_list = [
        # game/mechanism , learner
        ("configs_test/single_item/fpsb.yaml", "configs_test/learner/frank_wolfe.yaml")
    ]

    number_runs = 10
    learning = True
    simulation = True
    logging = True

    number_samples = int(2**22)
    save_strat = True
    save_init_strat = False
    round_decimal = 3

    main(
        path_exp,
        experiment_list,
        number_runs,
        learning,
        simulation,
        logging,
        number_samples,
        save_strat,
        save_init_strat,
        round_decimal,
    )
