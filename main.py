import argparse
from distributions import get_target_distribution
from divergence import get_divergence, divergence_minimization
from plot import plot
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
        "--lr",
        type=float,
        default="0.0001",
        help="learning rate",
    )
    return parser.parse_args()
if __name__ == "__main__":
    config = get_config()
    target_distributions = get_target_distribution([[0.0],[1.0]], [[0.1],[0.2]])
    divergence = get_divergence(config.distance_measure)
    result_distribution = divergence_minimization(target_distributions, divergence, config.lr, config.iter_num)
    plot(target_distributions, result_distribution, config.distance_measure)