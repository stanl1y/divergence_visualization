import torch
import matplotlib.pyplot as plt
import copy
from plot import plot

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


def fkl(target_distributions, result_distribution):
    fkl = 0
    for target_distribution in target_distributions:
        fkl += torch.mean(target_distribution.entropy())
        fkl -= torch.mean(
            result_distribution.log_prob(target_distribution.rsample(torch.Size([256])))
        )
    return fkl


def rkl(target_distributions, result_distribution):
    rkl = 0
    rkl += torch.mean(result_distribution.entropy())
    q_log_p = torch.hstack(
        [
            target_distribution.log_prob(result_distribution.rsample(torch.Size([256])))
            for target_distribution in target_distributions
        ]
    )
    q_log_p=torch.max(q_log_p,axis=1)[0]
    rkl -= torch.mean(q_log_p)
    return rkl


def jsd(target_distributions, result_distribution):
    return (
        fkl(target_distributions, result_distribution) / 2
        + rkl(target_distributions, result_distribution) / 2
    )


def w2(target_distributions, result_distribution):
    w2 = 0
    for target_distribution in target_distributions:
        w2 += torch.sum(
            (target_distribution.mean - result_distribution.mean) ** 2
        ) + torch.sum((target_distribution.stddev - result_distribution.stddev) ** 2)
    return w2


def divergence_minimization(target_distributions, divergence, lr, iter_num):
    # result_distribution = MultivariateNormal(torch.tensor([0.0, 0.0]), torch.eye(2))
    mean = torch.tensor([0.0], requires_grad=True)
    std = torch.tensor([1.0], requires_grad=True)
    optimizer = torch.optim.Adam([mean, std], lr=lr)
    loss_list = []
    best_loss = 10000
    best_mean = copy.deepcopy(mean)
    best_std = copy.deepcopy(std)
    frame_list=[]
    for i in range(iter_num):
        if std < 0.05:
            break
        result_distribution = torch.distributions.normal.Normal(mean, std)
        loss = divergence(target_distributions, result_distribution)
        if loss < best_loss:
            best_loss = loss
            best_mean = copy.deepcopy(mean)
            best_std = copy.deepcopy(std)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().numpy())
        if (i%100==0):
            img=plot(target_distributions, result_distribution,save_img=False)
            frame_list.append(img)
    # plot loss
    plot_loss(loss_list)
    return torch.distributions.normal.Normal(best_mean, best_std), frame_list


def plot_loss(loss_list):
    plt.plot(loss_list)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    # save
    plt.savefig("loss.png")
