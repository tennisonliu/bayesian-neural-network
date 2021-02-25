# Bayesian Neural Networks

PyTorch Implemenation of Bayesian Neural Networks trained using *Bayes by Backprop*.

Paper: Blundell, C., Cornebise, J., Kavukcuoglu, K. and Wierstra, D., 2015, June. 
        Weight uncertainty in neural network. 
        In International Conference on Machine Learning (pp. 1613-1622). PMLR.
       (http://proceedings.mlr.press/v37/blundell15.html)


## Running and Configuring Model:
Training and evaluation of the model are actioned through `main.py`, the main entry point. The BNNs and non-Bayesian MLPs are defined in `networks.py`. Functions required to run each experiment are included in `/regression`, `/reinforcement_learning` and `/classification`. Helper functions are included for loadin data `data_utils.py` and logging progress `utils.py`. 

At run-time, `main` reads from a model configuration set in `config.py`. The configurations required to replicate the results of the paper are presented as-is. 

*E.g.* To train any model:
```
python3 main.py --model [regression|classification|rl]
```
