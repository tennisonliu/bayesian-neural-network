'''
Helper function to generate plots.
'''
import os
import numpy as np
import matplotlib.pyplot as plt

def create_regression_plot(X_test, y_test, train_ds, model_name):
    ''' Create plot for regression task predictions '''
    if not os.path.exists('./graphs/'):
        os.makedirs('./graphs/')

    plt.style.use('seaborn-colorblind')
    fig = plt.figure(figsize=(9, 6))
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
    
    plt.scatter(train_ds.dataset.X, train_ds.dataset.y, label='Training data', marker='x', alpha=0.5, color='k', s=2)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylim([-1.5, 1.5])
    plt.xlim([-0.6, 1.4])

    plt.savefig(f'./graphs/regression_{model_name}.pdf', dpi=5000, bbox_inches='tight', pad_inches=0.1)