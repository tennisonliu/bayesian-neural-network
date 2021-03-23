# Bayesian Neural Networks

PyTorch Implemenation of Bayesian Neural Networks trained using *Bayes by Backprop* (BBB). For more information, see our poster:
[Bayesian Neural Network Presentation](graphs/final_graphs/BNN_Poster.pdf)

Paper: Blundell, C., Cornebise, J., Kavukcuoglu, K. and Wierstra, D., 2015, June. 
        Weight uncertainty in neural network. 
        In International Conference on Machine Learning (pp. 1613-1622). PMLR.
       (http://proceedings.mlr.press/v37/blundell15.html)

## Extensions to BBB
Additional approximate inference methods are implemented including:
* *Bayes by Backprop - Local Reparameterisation Trick* (https://arxiv.org/pdf/1506.02557.pdf): Samples pre-activations instead of weights for lower variance, faster computation and convergence.
* *Monte Carlo Dropout* (https://arxiv.org/pdf/1506.02142.pdf): Dropout during test-time to generate uncertainty measures, p=0.5.
* *Functional Variational Inference* (https://arxiv.org/pdf/1903.05779.pdf): Optimises ELBO defined on stochastic processes, *i.e.* distribution over functions.

## Running and Configuring Model:
Training and evaluation of the model are actioned through `main.py`, the main entry point. The BNNs and non-Bayesian MLPs are defined in `networks.py`. Functions required to run each experiment are included in 
* `/regression`, 
* `/reinforcement_learning`, &
* `/classification`.

Helper functions are included in `utils`. `data_utils.py` for loading data `logger_utils.py` for logging progress `plot_utils.py` for plotting and `load_model_utils.py` for loading trained models.. 

At run-time, `main` reads from a model configuration set in `config.py`. The configurations required to replicate the results of the paper are presented as-is. 

*e.g.* To train any model:
```
python3 main.py --model [regression|classification|rl]
```

## Analysis:
The scripts `weight_pruning.py` and `compute_ece.py` perform post-hoc analysis using saved models.
* `weight_pruning` 1) plots the distribution of weights, 2) computes SNR of BNNs, 3) evaluates performance on pruned weights.
* `compute_ece` 1) computes the expected calibration error (ECE) of trained model, 2) plots reliability diagram.

## TODO: 
* Refactor `reg_task.py`, `class_task` into base and derived classes / sort out inheritance.
