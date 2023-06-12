import torch
# from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


def get_target_distribution(means, stds):
    # means = [[0, 0], [1, 1], [2, 2]]
    # stds = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
    # if len(means[0]) > 1: # multivariate
    #     distribution = MultivariateNormal
    # else:
    distribution = Normal
    target_distributions = []
    for i in range(len(means)):
        target_distributions.append(
            distribution(torch.tensor(means[i]), torch.tensor(stds[i]))
        )
    return target_distributions
