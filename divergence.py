import torch
import matplotlib.pyplot as plt
import copy
from plot import plot_distribution, plot_loss


def get_divergence(distance_measure):
    if distance_measure == "fkl":
        return fkl
    elif distance_measure == "rkl":
        return rkl
    elif distance_measure == "jsd":
        return jsd
    elif distance_measure == "w2":
        return w2
    else:
        raise ValueError("distance measure not supported")


def fkl(target_distributions, result_distribution, sample_num):
    fkl = 0
    for target_distribution in target_distributions:
        fkl -= torch.mean(target_distribution.entropy())
        sampled_data = target_distribution.sample(torch.Size([sample_num]))
        fkl -= torch.mean(
            result_distribution.log_prob(sampled_data)
        )
    fkl /= len(target_distributions)
    return fkl


def rkl(target_distributions, result_distribution, sample_num):
    rkl = 0
    rkl -= torch.mean(result_distribution.entropy())
    sampled_data = result_distribution.rsample(torch.Size([sample_num]))
    q_log_p = torch.hstack(
        [
            target_distribution.log_prob(sampled_data)
            for target_distribution in target_distributions
        ]
    )
    q_log_p = torch.log(torch.mean(q_log_p.exp(), axis=1) + 0.00000001)
    rkl -= torch.mean(q_log_p)

    return rkl


def jsd(target_distributions, result_distribution, sample_num):
    return (
        fkl(target_distributions, result_distribution, sample_num) / 2
        + rkl(target_distributions, result_distribution, sample_num) / 2
    )


def w2(target_distributions, result_distribution, sample_num=None):
    w2 = 0
    for target_distribution in target_distributions:
        w2 += torch.sum(
            (target_distribution.mean - result_distribution.mean) ** 2
        ) + torch.sum((target_distribution.stddev - result_distribution.stddev) ** 2)
    return w2

def get_optimizer(params, lr, optimizer_type="adam"):
    if optimizer_type == "adam":
        return torch.optim.Adam(params, lr=lr)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(params, lr=lr)
    else:
        raise ValueError("optimizer not supported")


def divergence_minimization(target_distributions, distance_measure, lr, iter_num, sample_num, optimizer_type="adam"):
    # result_distribution = MultivariateNormal(torch.tensor([0.0, 0.0]), torch.eye(2))
    divergence = get_divergence(distance_measure)
    mean = torch.tensor([0.0], requires_grad=True)
    std = torch.tensor([1.0], requires_grad=True)
    optimizer = get_optimizer([mean, std], lr, optimizer_type)
    loss_list = []
    best_loss = float("inf")
    best_mean = copy.deepcopy(mean)
    best_std = copy.deepcopy(std)
    frame_list = []
    for i in range(iter_num):
        if std < 0.05:
            break
        result_distribution = torch.distributions.normal.Normal(mean, std)
        loss = divergence(target_distributions, result_distribution, sample_num)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().numpy())
        if len(loss_list) >= 100:
            smoothed_loss = sum(loss_list[-100:]) / 100
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                best_mean = copy.deepcopy(mean)
                best_std = copy.deepcopy(std)
        if i % 100 == 0:
            img = plot_distribution(target_distributions, result_distribution, save_img=False)
            frame_list.append(img)
    # plot loss
    plot_loss(loss_list,distance_measure,optimizer_type)
    return torch.distributions.normal.Normal(best_mean, best_std), frame_list



