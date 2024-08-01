# paths
PATH_TO_CONFIGS = "projects/contests/configs/"
PATH_TO_EXPERIMENTS = "experiments/contests/"
PATH_TO_RESULTS = "projects/contests/results/"

# parameter to run experiment
PARAM_LOGGING = {
    "path_experiment": PATH_TO_EXPERIMENTS,
    "save_strategy": True,
    "save_strategy_init": True,
    "save_image": True,
    "round_decimal": 5,
}
PARAM_COMPUTATION = {
    "active": True,
    "init_method": "random",
}
PARAM_SIMULATION = {
    "active": True,
    "number_samples": int(2**22),
}
PARAM_EVALUATION = {
    "active": False,
}
NUMBER_RUNS = 10

# parameter for evaluations and visualizations
ROUND_DECIMALS_TABLE = 3
DPI = 600
FORMAT = "pdf"
SAMPLES = 500
COLORS = ["#0096c4", "#f8766d", "#ffd782", "#9678aa"]
FONTSIZE_LABEL = 14
FONTSIZE_LEGEND = 15
FONTSIZE_TITLE = 16
