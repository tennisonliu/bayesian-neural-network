import os
import numpy as np
import matplotlib.pyplot as plt

def create_regression_plot(X_test, y_test, train_ds, model_name):
    ''' Create plot for regression task predictions '''
    if not os.path.exists('./graphs/'):
        os.makedirs('./graphs/')

    fig = plt.gcf()
    fig.set_size_inches(5, 3.5)
    plt.plot(X_test, np.median(y_test, axis=0), label='Median Posterior Predictive')
    
    # Range
    plt.fill_between(
        X_test.reshape(-1), 
        np.percentile(y_test, 0, axis=0), 
        np.percentile(y_test, 100, axis=0), 
        alpha = 0.2, color='orange', label='Range')
    
    # interquartile range
    plt.fill_between(
        X_test.reshape(-1), 
        np.percentile(y_test, 25, axis=0), 
        np.percentile(y_test, 75, axis=0), 
        alpha = 0.4, label='Interquartile Range')
    
    plt.legend()
    plt.scatter(train_ds.dataset.X, train_ds.dataset.y, label='Training data', marker='x', alpha=0.5, color='k', s=2)
    plt.ylim([-1.5, 1.5])
    plt.xlim([-0.6, 1.4])
    plt.savefig(f'./graphs/regression_{model_name}.png')
    plt.close()