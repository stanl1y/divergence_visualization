import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def plot(target_distributions, result_distribution, distance_measure):
    plt.figure()
    for target_distribution in target_distributions:
        mu = target_distribution.mean[0].detach().numpy()
        sigma = target_distribution.stddev[0].detach().numpy()
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        plt.plot(
            x,
            stats.norm.pdf(x, mu, sigma),
            label="target distribution",
        )
    mu = result_distribution.mean[0].detach().numpy()
    sigma = result_distribution.stddev[0].detach().numpy()
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(
        x,
        stats.norm.pdf(x, mu, sigma),
        label="result distribution",
    )
    plt.legend()
    plt.savefig(f"distribution_{distance_measure}.png")

