# Project: SODA

These are the numerical experiments for

>**Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging.**<br>
*Martin Bichler, Maximilian Fichtl, Matthias Oberlechner*<br>
Operations Research, 2023 (Forthcoming)

---
## Computation
To compute the strategies (learning) and the relevant metrics (simulation) run the following scripts. The results will be stored in `PATH_TO_EXPERIMENTS = "experiments/soda/"`.

**Exp 1 "interdependent" - Single-Object Auctions (Section 4.2)**<br>
Run experiments for single-item auctions with interdependencies, i.e., affiliated values auction and common value auction. <br>
Number of experiments: 10 (x 10 runs), runtime incl. simulation: ~?min</sub>
```bash
python projects/soda/computation/run_interdependent.py
```

**Exp 2 "llg" - Combinatorial Auctions in the Local-Local-Global Model (Section 4.3.)**<br>
Number of experiments: 45 (x 10 runs ) + 3 (x 1 run), runtime incl. simulation: 3.5h 
```bash
python projects/soda/computation/run_llg.py
```

**Exp 3  "split_award" - Combinatorial Split-Award Auction (Section 4.4.)** <br>
Number of experiments: 10 (x 10 runs ), runtime incl. simulation: 9.5h
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
## Evaluation
To create the tables from the numerical experiments, i.e., Table 3-10 run
```bash
python projects/soda/evaluation/generate_tables.py
```
To create the figures from the numerical experiments, i.e., Figure 2-7 run
```bash
python projects/soda/evaluation/generate_plots.py
```
The results are stored in separate directory in `PATH_TO_RESULTS = "experiments/soda/"`.