import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from io import BytesIO


def plot_distribution(
    target_distributions, result_distribution, distance_measure=None, optimizer_type=None, save_img=True
):
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
    plt.xlim([-4, 4])
    if save_img:
        plt.savefig(f"result/distribution_{distance_measure}_{optimizer_type}.png")
        plt.close()
    else:
        # return image (numpy array)
        buffer_ = BytesIO()
        plt.savefig(buffer_, format="png")
        buffer_.seek(0)
        img = plt.imread(buffer_)
        plt.close()
        img *= 255
        img = img.astype(np.uint8)
        return img


def plot_loss(loss_list, distance_measure, optimizer_type):
    loss_list=loss_list[::20]
    smoothed_loss = np.convolve(loss_list, np.ones(100)/100, mode='valid')
    plt.plot(smoothed_loss)

    plt.xlabel("iteration")
    plt.ylabel("loss")
    # save
    plt.savefig(f"result/loss_{distance_measure}_{optimizer_type}.png")
