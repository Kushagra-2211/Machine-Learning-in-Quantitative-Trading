# Machine Learning in Quantitative Trading Strategies: Pairs Trading and Momentum Trading in Equity and Cryptocurrency Markets

Code accompanying an MSc Finance Analytics dissertation (King's College London). Benchmarks a classical, rules-based trading strategy against five machine learning models (Random Forest, SVM, Neural Network, LSTM, Gradient Boosting) under an identical walk-forward, bootstrap-and-Wilcoxon testing protocol, across two strategy/asset-class pairings:

- **`pairs_trading/`** — cointegration-based pairs trading on U.S. equities.
- **`momentum_trading/`** — moving-average momentum trading on cryptocurrencies.

Each package follows the same structure: strategy specification, ML model implementations, walk-forward retraining, regime classification (Kaufman's Efficiency Ratio), and performance/statistical-testing utilities (`performance.py`), orchestrated through `main.py`.

Raw data, cached downloads, and run outputs are not included in this repository.
