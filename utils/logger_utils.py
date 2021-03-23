'''
Helper function to log progress to Tensorboard
'''
import torch

def write_bandit_action(writer, action_class, step):
    ''' Logging tool to record bandit actions '''
    writer.add_scalar('actions/tp', action_class[0], step)
    writer.add_scalar('actions/fp', action_class[1], step)
    writer.add_scalar('actions/tn', action_class[2], step)
    writer.add_scalar('actions/fn', action_class[3], step)

def write_weight_histograms(writer, net, step):
    ''' Logging tool for BNN '''
    writer.add_histogram('histogram/w1_mu', net.l1.weight_mu, step)
    writer.add_histogram('histogram/w1_rho', torch.log1p(torch.exp(net.l1.weight_rho)), step)
    writer.add_histogram('histogram/w2_mu', net.l2.weight_mu, step)
    writer.add_histogram('histogram/w2_rho', torch.log1p(torch.exp(net.l2.weight_rho)), step)
    writer.add_histogram('histogram/w3_mu', net.l3.weight_mu, step)
    writer.add_histogram('histogram/w3_rho', torch.log1p(torch.exp(net.l3.weight_rho)), step)
    writer.add_histogram('histogram/b1_mu', net.l1.bias_mu, step)
    writer.add_histogram('histogram/b1_rho', torch.log1p(torch.exp(net.l1.bias_rho)), step)
    writer.add_histogram('histogram/b2_mu', net.l2.bias_mu, step)
    writer.add_histogram('histogram/b2_rho', torch.log1p(torch.exp(net.l2.bias_rho)), step)
    writer.add_histogram('histogram/b3_mu', net.l3.bias_mu, step)
    writer.add_histogram('histogram/b3_rho', torch.log1p(torch.exp(net.l3.bias_rho)), step)

def write_loss_scalars(writer, loss, step):
    ''' Logging tool for BNN '''
    if len(loss) == 4:
        writer.add_scalar('logs/loss', loss[0], step)
        writer.add_scalar('logs/complexity_cost', loss[2]-loss[1], step)
        writer.add_scalar('logs/log_prior', loss[1], step)
        writer.add_scalar('logs/log_variational_posterior', loss[2], step)
        writer.add_scalar('logs/negative_log_likelihood', loss[3], step)
    else:
        writer.add_scalar('logs/loss', loss[0], step)
        writer.add_scalar('logs/complexity_cost', loss[1], step)
        writer.add_scalar('logs/negative_log_likelihood', loss[2], step)

def write_loss(writer, loss, step):
    ''' Logging tool MLP '''
    writer.add_scalar('logs/loss', loss, step)

def write_acc(writer, acc, step):
    ''' Logging tool classification '''
    writer.add_scalar('logs/acc', acc, step)