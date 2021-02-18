import pandas as pd
from sklearn import preprocessing
from config import *

def read_data():
    '''
    Function to read in data, transform context and label to one-hot encoded vectors
    '''
    df = pd.read_csv(Config.data_dir, sep=',', header=None, error_bad_lines=False, warn_bad_lines=True, low_memory=False)
    df.columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment',
         'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root',
         'stalk-surf-above-ring','stalk-surf-below-ring','stalk-color-above-ring','stalk-color-below-ring',
         'veil-type','veil-color','ring-number','ring-type','spore-color','population','habitat']
    X = pd.DataFrame(df, columns=df.columns[1:len(df.columns)], index=df.index)
    Y = df['class']

    # transform to one-hot encoding
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(Y)
    Y_ = label_encoder.transform(Y)
    X_ = X.copy()
    for feature in X.columns:
        label_encoder.fit(X[feature])
        X_[feature] = label_encoder.transform(X[feature])

    oh_encoder = preprocessing.OneHotEncoder()
    oh_encoder.fit(X_)
    X_ = oh_encoder.transform(X_).toarray()

    return X_, Y_

def write_weight_histograms(writer, net, step):
    ''' Logging tool for BNN Bandit '''
    
    writer.add_histogram('histogram/w1_mu', net.l1.weight_mu,step)
    writer.add_histogram('histogram/w1_rho', torch.log1p(torch.exp(net.l1.weight_rho)),step)
    writer.add_histogram('histogram/w2_mu', net.l2.weight_mu,step)
    writer.add_histogram('histogram/w2_rho', torch.log1p(torch.exp(net.l2.weight_rho)),step)
    writer.add_histogram('histogram/w3_mu', net.l3.weight_mu,step)
    writer.add_histogram('histogram/w3_rho', torch.log1p(torch.exp(net.l3.weight_rho)),step)
    writer.add_histogram('histogram/b1_mu', net.l1.bias_mu,step)
    writer.add_histogram('histogram/b1_rho', torch.log1p(torch.exp(net.l1.bias_rho)),step)
    writer.add_histogram('histogram/b2_mu', net.l2.bias_mu,step)
    writer.add_histogram('histogram/b2_rho', torch.log1p(torch.exp(net.l2.bias_rho)),step)
    writer.add_histogram('histogram/b3_mu', net.l3.bias_mu,step)
    writer.add_histogram('histogram/b3_rho', torch.log1p(torch.exp(net.l3.bias_rho)),step)

def write_loss_scalars(writer, loss, regret, step):
    ''' Logging tool for BNN Bandit '''
    writer.add_scalar('logs/loss', loss[0], step)
    writer.add_scalar('logs/complexity_cost', loss[2]-loss[1], step)
    writer.add_scalar('logs/log_prior', loss[1], step)
    writer.add_scalar('logs/log_variational_posterior', loss[2], step)
    writer.add_scalar('logs/negative_log_likelihood', loss[3], step)
    writer.add_scalar('logs/cumulative_regret', regret, step)

def write_loss(writer, loss, regret, step):
    ''' Logging tool for e-greedy MLP '''
    writer.add_scalar('logs/loss', loss, step)
    writer.add_scalar('logs/cumulative_regret', regret, step)