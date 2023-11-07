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


def get_runtimes(path_to_experiments: str, experiment_tag: str) -> pd.DataFrame:
    """Get runtimes for logging of learning

    Args:
        path_to_experiments (str): path to respective subdirectory of experiments
        experiment_tag (str): experiment tag, i.e., subdirectory of log directory

    Returns:
        pd.DataFrame: df containing the runtimes
    """

    # import file
    file_log_learn = os.path.join(
        path_to_experiments, "log", experiment_tag, "log_learn.csv"
    )
    df = pd.read_csv(file_log_learn)

    # get relevant columns and delete duplicates (due to several agents)
    cols = ["mechanism", "setting", "learner", "run", "time_init", "time_run"]
    df = df[cols].drop_duplicates().reset_index(drop=True)

    # get runtimes (min, max)
    df["time_total"] = df["time_init"] + df["time_run"]
    df = df.groupby(["setting", "mechanism", "learner"]).agg(
        {"time_init": "first", "time_run": ["min", "max"], "time_total": ["min", "max"]}
    )
    df = df.reset_index()

    # create text for table
    df["time"] = [
        time_to_str(t_min=df["time_total"]["min"][i], t_max=df["time_total"]["max"][i])
        for i in df.index
    ]
    return df


def time_to_str(t_min, t_max) -> str:
    """Method that brings runtime in format for table."""
    if t_max < 100:
        if t_min < 2:
            return f"{t_min:.1f}-{t_max:.1f} s"
        else:
            return f"{t_min:.0f}-{t_max:.0f} s"
    else:
        if t_min < 300:
            return f"{t_min/60:.1f}-{np.ceil(t_max/60):.1f} min"
        else:
            return f"{t_min/60:.0f}-{np.ceil(t_max/60):.0f} min"
