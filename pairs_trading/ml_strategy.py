"""
DEPRECATED -- kept only so nothing breaks if something still imports this
file by its old name. The single "ml_strategy.py" has been split into:

  ml_common.py        - shared feature engineering / signal logic
  ml_random_forest.py  - Random Forest (this used to be ml_strategy.py)
  ml_svm.py            - Support Vector Regression
  ml_neural_net.py      - feedforward Neural Network (PyTorch)
  ml_lstm.py            - LSTM (PyTorch)

Use ml_random_forest.py directly instead of this file going forward.
"""

from ml_random_forest import run_strategy, train_model  # noqa: F401
