# -------------------------------------------------------------------------------------------------------------------- #
#                                  SCRIPT WITH USEFUL METHODS FOR EVALUATION etc                                       #
# -------------------------------------------------------------------------------------------------------------------- #
import os

from soda.strategy import Strategy
from soda.util.config import Config


def get_results(
    config_game: dict,
    config_learner: dict,
    path_to_configs: str,
    path_to_experiment: str,
    experiment_tag: str,
    run: int = 0,
):
    """Import computed strategies for a given experiment

    Args:
        config_game (dict): config file for game
        config_learner (dict): config file for learner
        path_to_configs (str): path to config files
        path_to_experiment (str): path to experiments
        experiment_tag (str): experiment tag i.e. subdirectory in experiments
        run (int, optional): Run of experiment. Defaults to 0.
    """
    config_game = os.path.join(path_to_configs, "game", config_game)
    config_learner = os.path.join(path_to_configs, "learner", config_learner)

    config = Config(config_game, config_learner)
    game, learner = config.create_setting()
    strategies = config.create_strategies(game)

    learner_name = os.path.basename(config_learner).replace(".yaml", "")
    game_name = os.path.basename(config_game).replace(".yaml", "")
    name = f"{learner_name}_{game_name}_run_{run}"
    path = os.path.join(path_to_experiment, "strategies", experiment_tag)

    for i in strategies:
        strategies[i] = Strategy(i, game)
        strategies[i].load(name, path, load_init=False)

    return game, learner, strategies
