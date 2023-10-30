# Project: SODA

These are the numerical experiments for

>**Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging.**<br>
*Martin Bichler, Maximilian Fichtl, Matthias Oberlechner*<br>
24th ACM Conference on Economics and Computation (ACM-EC), 2023

---

### Computation
To compute the strategies (learning) and evaluate them (simulation) run the following scripts.
The results will be stored in `experiments/soda/`

**Exp 1 "interdependent" - Single-Object Auctions (Section 4.2)**<br>
Run experiments for single-item auctions with interdependencies, i.e., affiliated values auction and common value auction.
```bash
python projects/soda/computation/run_interdependent.py
```
<sub>Number of settings: 10, Runtime: ~?min</sub>

**Exp 2 "llg" - Combinatorial Auctions in the Local-Local-Global Model (Section 4.3.)**<br>
```bash
python projects/soda/computation/run_llg.py
```
<sub>Number of settings: x, Runtime: ~?min</sub>

**Exp 3  "split_award" - Combinatorial Split-Award Auction (Section 4.4.)** <br>
Number of settings: 10 (x 10 runs ), runtime incl. simulation: 
```bash
python projects/soda/simulation/run_split_award.py
```

**Exp 4 "risk" - Single-Object Auctions with Risk-Averse Bidders (Section 4.5.)** <br>
Run experiment for first-price sealed-bid auction and all-pay auction with different levels of risk aversion. <br>
Number of settings: 36 (x 10 runs ), runtime incl. simulation: 
```bash
python projects/soda/simulation/run_risk.py
```

**Exp 5 - Tullock Contests (Section 4.6.)**<br>
```bash
python
```
<sub>Number of settings: x, Runtime: ~?min</sub>