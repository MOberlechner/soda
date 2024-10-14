# Project: Ad-Auctions
These are the numerical experiments (with SODA) for:

>**Low Revenue in Display Ad Auctions: Algorithmic Collusion vs. Non-Quasilinear Preferences**<br>
*Martin Bichler, Alok Gupta, Laura Mathews, Matthias Oberlechner<br> 2023, Working Paper*

To reproduce the results (with SODA) from the paper above, run the scripts as described below:

---
The parameters for the experiments can be found in [config_exp](config_exp.py):
- the results will be stored in<br> `PATH_TO_EXPERIMENTS = "experiments/ad_auctions/"` 
- the config files can be found in<br> `PATH_TO_CONFIGS = "projects/ad_auction/configs/"`
- plots will be stored in<br> `PATH_SAVE = "projects/ad_auctions/results/"`

## Exp 1 - Revenue for FPSB and SPSB with QL, ROI, and ROSB
*Number of settings: 48 (10 runs each), Runtime: 80 min*  <br>
**Computation**<br>
Experiments to compute BNE for different utility models and compare expected revenue.
We vary the number of agents (2, 3, 5, and 10) and the prior (uniform and truncated gaussian).
```bash
python projects/ad_auctions/computation/run_revenue.py
```
**Evaluation**<br>
We create 3 figures (Figure 3) where we compare the difference in revenue between first- and second-price auctions, and one table (Table 2) in which we report metrics where we compare the computed strategies to known analytical BNE.
Additional plots that visualize the computed strategies are also created.
```bash
python projects/ad_auctions/evaluation/plot_revenue.py
python projects/ad_auctions/evaluation/table_revenue.py
```

## Exp 2 - Asymmetric Utility Functions
*Number of settings: 12 (10 runs each), Runtime: 12 min*  <br>
**Computation**<br>
Experiments to compute BNE for different utility models and compare expected revenue.
We vary the number of agents (2, 3, 5, and 10) and the prior (uniform and truncated gaussian).
```bash
python projects/ad_auctions/computation/run_revenue.py
```
**Evaluation**<br>
We create 3 figures (Figure 3) where we compare the difference in revenue between first- and second-price auctions, and one table (Table 2) in which we report metrics where we compare the computed strategies to known analytical BNE.
Additional plots that visualize the computed strategies are also created.
```bash
python projects/ad_auctions/evaluation/plot_revenue.py
python projects/ad_auctions/evaluation/table_revenue.py
```


## Exp 3 - Budget Constraints
*Number of settings: 12 (10 runs each), Runtime: 5 min* 

**Computation**<br>
For better comparison, we consider budget constraints via log-barrier function for QL, ROI, and ROS. 
```bash
python projects/ad_auctions/computation/run_revenue_budget.py
```

**Evaluation**<br>
Given the computed strategies, we compare the expected revenue of first- and second-price auctions for these budget constraints.
```bash
python projects/ad_auctions/evaluation/plot_revenue_budget.py
```

## Exp 4 - ROIS: Convex Combination of ROI and ROS
*Number of settings: 10 (10 runs each), Runtime: 4 min*

**Compuation**<br>
In this experiments we consider an extension of the ROI and ROS utility functions, by considering convex combinations thereof. We compute the equilibrium strategies for different combinations.
```bash
python projects/ad_auctions/computation/run_revenue_rois.py
```

**Evaluation**<br>
We create on one plot (Figure 4 a) that shows the expected revenue for the first- and second-price auctions in equilibrium given the utility functions.
```bash
python projects/ad_auctions/evaluation/plot_revenue_rois.py
```

## Exp 5 - Baseline for Bandits
*Number of settings: 6 (1 run each), Runtime: 20s*

**Computation**<br>
Experiments with a low discretization (n=21) as a baseline to compare to results using bandit algorithms. Here we only compute the strategies.
```bash
python projects/ad_auctions/computation/run_baseline.py
```

**Evaluation**<br>
Compute the expected revenue in the discretized setting.
```bash
python projects/ad_auctions/evaluation/table_baseline.py 
```

---
## Run all experiments:<br>

**Computation - create data (~1.5h)**
```
python projects/ad_auctions/computation/run_revenue.py | python projects/ad_auctions/computation/run_revenue_rois.py | python projects/ad_auctions/computation/run_revenue_budget.py | python projects/ad_auctions/computation/run_baseline.py
```
**Evaluation - create plots and tables**
```
python projects/ad_auctions/evaluation/plot_revenue.py | python projects/ad_auctions/evaluation/table_revenue.py | python projects/ad_auctions/evaluation/plot_revenue_rois.py | python projects/ad_auctions/evaluation/plot_revenue_budget.py
```
---
*For questions, please contact: matthias.oberlechner@tum.de.*