# Project: SODA

These are the numerical experiments for

>**Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging.**<br>
*Martin Bichler, Maximilian Fichtl, Matthias Oberlechner*<br>
24th ACM Conference on Economics and Computation (ACM-EC), 2023

---

### Simulation
To compute the strategies (learning) and evaluate them (simulation) run the following scripts.
The results will be stored in `experiments/soda/`

**Exp 1 - Single-Object Auctions (Section 4.2)**<br>
*Number of settings: x, Runtime: ~?min*  <br>

**Exp 2 - Combinatorial Auctions in the Local-Local-Global Model (Section 4.3.)**<br>
*Number of settings: x, Runtime: ~?min*  <br>

**Exp 3 - Combinatorial Split-Award Auction (Section 4.4.)** <br>
*Number of settings: x, Runtime: ~?min*  <br>

**Exp 4 - Single-Object Auctions with Risk-Averse Bidders (Section 4.5.)** <br>
*Number of settings: x, Runtime: ~?min*  <br>
Run experiment for first-price sealed-bid auction and all-pay auction with different levels of risk aversion.
```bash
python projects/soda/simulation/run_risk.py
```
**Exp 5 - Tullock Contests (Section 4.6.)**<br>