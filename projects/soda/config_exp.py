# paths
PATH_TO_CONFIGS = "projects/soda/configs/"
PATH_TO_EXPERIMENTS = "experiments/soda_test/"
PATH_TO_RESULTS = "projects/soda/results/"

# parameter to run experiments
PARAM_LOGGING = {
    "path_experiment": PATH_TO_EXPERIMENTS,
    "save_strategy": True,
    "save_strategy_init": False,
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
SAMPLES = 150
MARKER = ["o", "s", "v"]
MARKER_SIZE = 30
COLORS = ["#003f5c", "#bc5090", "#ffa600"]
FONTSIZE_LABEL = 14
FONTSIZE_LEGEND = 15
FONTSIZE_TITLE = 16
