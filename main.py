import argparse
from distributions import get_target_distribution
from divergence import get_divergence, divergence_minimization
from plot import plot_distribution
import imageio


def get_config():
    parser = argparse.ArgumentParser(description="divergence visualization")
    parser.add_argument(
        "--distance_measure",
        type=str,
        default="fkl",
        help="which distance measure to use, fkl,rkl,jsd,w2",
    )
    parser.add_argument(
        "--iter_num",
        type=int,
        default="20000",
        help="number of training iteration",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default="10",
        help="number of data point sampled from distribution",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default="0.0001",
        help="learning rate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()
    target_distributions = get_target_distribution()
    result_distribution, frame_list = divergence_minimization(
        target_distributions, config.distance_measure, config.lr, config.iter_num, config.sample_num
    )
    plot_distribution(target_distributions, result_distribution, config.distance_measure)

    imageio.mimsave(
        f"result/distribution_{config.distance_measure}.gif",
        frame_list,
        loop=0,
        duration=100,  # output gif  # array of input frames
    )  # optional: frames per second
