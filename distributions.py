import torch
# from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


def get_target_distribution():
    with open("target_distribution_config", "r") as f:
        f.readline() # skip first line
        lines = f.readlines()
        means = []
        stds = []
        for line in lines:
            mean, std = line.split(",")
            means.append([float(mean)])
            stds.append([float(std)])
    distribution = Normal
    target_distributions = []
    for i in range(len(means)):
        target_distributions.append(
            distribution(torch.tensor(means[i]), torch.tensor(stds[i]))
        )
    return target_distributions
