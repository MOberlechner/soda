import os

import numpy as np
import pandas as pd

from projects.contests.config_exp import PATH_TO_EXPERIMENTS, PATH_TO_RESULTS
from soda.util.evaluation import create_table

if __name__ == "__main__":

    experiment_tag = "crowdsourcing"
    path_save = os.path.join(PATH_TO_RESULTS, experiment_tag)
    os.makedirs(PATH_TO_RESULTS, exist_ok=True)

    # Save csv
    df = create_table(PATH_TO_EXPERIMENTS, experiment_tag)
    df.to_csv(os.path.join(PATH_TO_RESULTS, "table_crowdsourcing.csv"), index=False)

    # Print Latex Table
    cols = ["setting", "util_loss_discr", "time", "l2_norm", "util_loss"]
    cols_format = ["l", "r", "r", "c", "c"]
    table = df[cols].style.to_latex(column_format=cols_format)
    print(f"\nTABLE CROWDSOURCING\n\n{table}\n")
