# Project: Ad-Auctions
These are the numerical experiments (with SODA) for:

>**Low Revenue in Display Ad Auctions: Algorithmic Collusion vs. Non-Quasilinear Preferences**<br>
*Martin Bichler, Alok Gupta, Laura Mathews, Matthias Oberlechner<br> 2023, Working Paper*

---

## Computation
- the results will be stored in<br> `PATH_TO_EXPERIMENTS = "experiments/ad_auctions/"` 
- the config files can be found in<br> `PATH_TO_CONFIGS = "projects/ad_auction/configs/"`

To compute the strategies (learning) and evaluate them (simulation) run the following scripts (from the main directory of the repo).

**Exp 1 - FPSB and SPSB for different utility models (large)**<br>
*Number of settings: 16 (10 runs each), Runtime: 23 min*  <br>
Experiments to compute BNE for different utility models and compare expected revenue.
This settings uniform prior (2-3 agents) and a truncated gaussian prior (2 agents).
```bash
python projects/soda/computation/run_revenue.py
```

**Exp 2 - FPSB and SPSB for different utility models (baseline)**<br>
*Number of settings: 6, Runtime: 20s*  <br>
Experiments with a low discretization (n=21) as a baseline to compare to results using bandit algorithms. Here we only compute the strategies.
```bash
python projects/soda/computation/run_baseline.py
```

## Evaluation
After the strategies are computed and saved (together with log files) in the `PATH_TO_EXPERIMENT` directory, we can evaluate our results.

**Exp 1 - FPSB and SPSB for different utility models (large)**<br>
To create a plot with the BNE strategies and the expected revenue run
```bash
python projects/soda/evaluation/plot_revenue.py
```
The table in the appendix (and more) where we compare the computed BNE to the analytical ones is created with
```bash
python projects/soda/evaluation/table_revenue.py
```

**Exp 2 - FPSB and SPSB for different utility models (baseline)**<br>
We compute the expected revenue in the discretized setting to serve as a baseline for the bandit learner.
```bash
python projects/soda/evaluation/table_baseline.py
```
---
*contact: matthias.oberlechner@tum.de*