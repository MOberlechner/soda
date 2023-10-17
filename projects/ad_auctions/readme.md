# Project: Ad-Auctions
These are the numerical experiments for:

>**Low Revenue in Display Ad Auctions: Algorithmic Collusion vs. Non-Quasilinear Preferences**<br>
*Martin Bichler, Alok Gupta, Laura Mathews, Matthias Oberlechner (2023, Working Paper)*

---

### Simulation
- the results will be stored in<br> `PATH_TO_EXPERIMENTS = "experiments/ad_auctions/"` 
- the config files can be found in<br> `PATH_TO_CONFIGS = "projects/ad-auction/configs/"`

To compute the strategies (learning) and evaluate them (simulation) run the following scripts (from the main directory of the repo).

**Exp 1 - FPSB and SPSB for different utility models (large)**<br>
*Number of settings: 10 (10 runs each), Runtime: 18 min*  <br>
Experiments to compute BNE for different utility models and compare expected revenue
```bash
python projects/soda/simulation/run_revenue.py
```

**Exp 2 - FPSB and SPSB for different utility models (large)**<br>
*Number of settings: x, Runtime: ~?min*  <br>
Experiments with a low discretization (n=21) as a baseline to compare to results using bandit algorithms. Here we only compute the strategies.
```bash
python projects/soda/simulation/run_baseline.py
```
