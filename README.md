# Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging (SODA)
The code was created for the following paper [[arxiv]](https://arxiv.org/abs/2208.02036):
```
@misc{bichler2023soda,
      title={Computing Bayes Nash Equilibrium Strategies in Auction Games 
      via Simultaneous Online Dual Averaging}, 
      author={Martin Bichler and Maximilian Fichtl and Matthias Oberlechner},  
      year={2023},  
      eprint={2208.02036}, 
      archivePrefix={arXiv},  
      primaryClass={cs.GT}  
}
```

## Experiments from Publications
You can use the config files to reproduce the experiments from the publications

- **Computing Bayes Nash Equilibrium Strategies in Auction Games via Simultaneous Online Dual Averaging** 
<br> Martin Bichler, Maximilian Fichtl, Matthias Oberlechner [[arXiv:2208.02036](https://arxiv.org/abs/2208.02036)]
<br> >> run experiments in [experiments/paper_soda_arxiv](https://github.com/MOberlechner/soda/tree/main/experiments/paper_soda_arxiv)
- **Learning equilibrium in bilateral bargaining games**
<br> Martin Bichler, Nils Kohring, Matthias Oberlechner, Fabian R Pieroth [[EJOR](https://www.sciencedirect.com/science/article/abs/pii/S0377221722009705)]
<br> ... to be included ..

## What is implemented?

We focus on incomplete-information (Bayesian) games with continuous type and action space. 
By discretizing the type and action space, and using distributional strategies apply standard (gradient-based) learning algorithms to approximate Bayes-Nash equilibria (BNE) of given mechanisms.

#### Mechanisms

- Single-Item Auctions (first- and second-price auctions with risk-aversion, ...)
- All-Pay Auctions (first- and second-price, i.e. War of attrition)
- LLG-Auction (small combinatorial auction with 2 items, 2 local bidders and 1 global bidder)
- Split-Award Auction (procurement auction with 2 agent)
- Tullock Contests

#### Learning Algorithms

- Dual Averaging (Gradient ascent with lazy projection, exponentiated gradient ascent)
- Mirror Ascent (projected gradientascent, exponentiated gradient ascent)
- Frank-Wolfe Algorithm
- Fictitious PLay
