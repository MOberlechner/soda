# paths
PATH_TO_CONFIGS = "projects/ad_auctions/configs/"
PATH_TO_EXPERIMENTS = "experiments/ad_auctions_test/"
PATH_TO_RESULTS = "projects/ad_auctions/results_test/"

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
NUMBER_RUNS = 10

# parameter for visualizations
FONTSIZE_TITLE = 14
FONTSIZE_LEGEND = 13
FONTSIZE_LABEL = 13
COLORS = ["#003f5c", "#ffa600", "#bc5090"]
