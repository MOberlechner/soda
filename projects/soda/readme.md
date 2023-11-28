# Project: SODA

>**Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging.**<br>
*Martin Bichler, Maximilian Fichtl, Matthias Oberlechner*<br>
Operations Research, 2023 (Forthcoming)

To reproduce the numerical experiments from the paper:
1. run the scripts in [projects/soda/computation](./computation) to compute the strategies and respective metrics. 
2. create the plots and tables from the paper by running the scripts in  [projects/soda/computation](./evaluation).

All parameters for the specific experiments can be found in [projects/soda/configs](./configs). More general parameters for the experiments and visualizations can be found in [projects/soda/config_exp](./config_exp.py).

---
### **Computation**
To compute the strategies (learning) and the relevant metrics (simulation) run the following scripts. The results will be stored in `PATH_TO_EXPERIMENTS = "experiments/soda/"`.

**Example**<br>
Number of experiments: 1, runtime: 1s
```bash
python projects/soda/computation/run_example.py
```

**Exp 1 "interdependent" - Single-Object Auctions (Section 4.2)**<br>
Number of experiments: 10 (x 10 runs), runtime incl. simulation: 6.7h<br>
```bash
python projects/soda/computation/run_interdependent.py
```
Run experiments for single-item auctions with interdependencies, i.e., affiliated values auction and common value auction. <br>

**Exp 2 "llg" - Combinatorial Auctions in the Local-Local-Global Model (Section 4.3.)**<br>
Number of experiments: 45 (x 10 runs ) + 3 (x 1 run), runtime incl. simulation: 3.5h 
```bash
python projects/soda/computation/run_llg.py
```

**Exp 3  "split_award" - Combinatorial Split-Award Auction (Section 4.4.)** <br>
Number of experiments: 10 (x 10 runs ), runtime incl. simulation: 11h
```bash
python projects/soda/computation/run_split_award.py
```

**Exp 4 "risk" - Single-Object Auctions with Risk-Averse Bidders (Section 4.5.)** <br>
Run experiment for first-price sealed-bid auction and all-pay auction with different levels of risk aversion. <br>
Number of experiments: 36 (x 10 runs ), runtime incl. simulation: 16min
```bash
python projects/soda/computation/run_risk.py
```

**Exp 5 - Tullock Contests (Section 4.6.)**<br>
Number of experiments: 10 (x 1 run ), runtime incl. simulation: 20s
```bash
python projects/soda/computation/run_contests.py
```

**Exp 6 - Discretizations (Appendix)**<br>
Number of experiments: 17 (x 10 run ), runtime incl. simulation: 50min
```bash
python projects/soda/computation/run_discretization.py
```

### **Evaluation**
To create the tables and the plots run the following scripts.
The results are stored in separate directories in `PATH_TO_EXPERIMENTS`.
```bash
python projects/soda/evaluation/generate_tables.py
python projects/soda/evaluation/generate_plots.py
python projects/soda/evaluation/generate_plots_appendix.py
```

