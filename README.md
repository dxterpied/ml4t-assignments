# [CS-7646-O1] Machine Learning for Trading: Assignments

The following projects are included in this repository:

## Assess Portfolio

In this project, I used Python Pandas to read stock data, compute different statistics and metrics and compare various portfolios. The metrics that were computed are as follows:

* Cumulative return
* Average Daily return
* Standard deviation of daily returns
* Sharpe ratio of the overall portfolio
* Ending value of the portfolio

## Optimize Something

In this project, I implemented a portfolio optimizer, that is, I found how much of a portfolio's fund should be allocated to each stock so as to optimize its performance. The optimization objective was to maximize the Sharpe Ratio, and it was modeled as a simple **linear program**. My optimizer was able to find an allocation that substantially beat the market. A graph can be seen [here](https://github.gatech.edu/arx3/ml4t-assignments/blob/master/optimize_something/plot.pdf).

## Marketsim

As the name implies, in this project I created a market simulator that accepts trading orders and keeps track of a portfolio's value over time and then assesses the performance of that portfolio.

## Defeat Learners

In this project, I generated data that I believed would work better for one type of Machine Learning model than another with the objective of assessing the understanding of the strengths and weaknesses of models. The two learned that were used in this project are a **Decision Tree** and a **Linear Regression** model. To solve this problem, I generated a completely **linear** dataset which, of course, gave the advantage to the Linear Regression model, and a **higher order polynomial** dataset which throws off the Linear Regression model and for which the Decision Tree has a better chance of manipulating correctly.

## Assess Learners

In this project, I implemented and evaluated three types of **tree-based learning** algorithms: **Decision Tree**, **Random Tree** and a **Bagged Tree**. These algorithms were compared based on their sensitivity to **overfitting**, their **generalization** power and their overall **correlation** between the predicted and true values. The complete report can be found [here](https://github.gatech.edu/arx3/ml4t-assignments/blob/master/assess_learners/report/report.pdf).

## Q-Learning Robot

This project served as an introduction to Reinforcement Learning. Here, I implemented the classic tabular **Q-Learning** and **Dyna-Q** algorithms to the Reinforcement Learning problem of navigating in a 2D grid world. The idea was to work on an easy problem before applying Q-Learning to the harder problem of trading.

## Manual Strategy

In this project, I developed a **trading strategy** using my own intuition and **technical indicators**, and tested it againts `$JPM` stock using the market simulator implemented previously. The technical indicators used are as follows:

* Momentum
* Price/SMA Ratio
* Bollinger Bands
* Money Flow Index

My rule-based strategy was compared against the benchmark of **holding** a `LONG` position for the stock until the end of the period. For the **in-sample** data, my strategy was able to achieve a cummulative return of over **36%** versus the benchmark return of **1.2%**. On the other hand, for the **out-of-sample** data, my strategy achieved a cummulative return of around **11%** versus the benchmark return of less than **1%**. Not bad for my first trading strategy! To full report can be found [here](https://github.gatech.edu/arx3/ml4t-assignments/blob/master/manual_strategy/report/report.pdf).

## Strategy Learner

For the final project, I implemented a ML-based program that learned the best trading strategy without any manual rules. Because a trading strategy can be seen as a **trading policy**, it was natural to model this problem as a **Reinforcement Learning** task with the following mapping:

* **States:** The technical indicators developed in the previous project.
* **Actions:** `LONG`, `SHORT` or `CASH` (i.e. closing any open position).
* **Reward:** Daily return.

Because we were limited by the concepts learned in this class, I discretized all of the technical indicators into *buckets* in order to apply the tabular **Q-Learning** algorithm that was developed in the Q-Learning Robot project. Nevertheless, even with discretization, my Q-Learner was able to find an optimal strategy that beat both the benchmark and my previous manual strategy. The complete report can be found [here](https://github.gatech.edu/arx3/ml4t-assignments/blob/master/strategy_learner/report/report.pdf).

## Resources

Course website: http://quantsoftware.gatech.edu/CS7646_Fall_2017

Information on cloning this repository and using the autograder on buffet0x servers: http://quantsoftware.gatech.edu/ML4T_Software_Setup
