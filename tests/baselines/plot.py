import matplotlib.pyplot as plt
import numpy as np

from tests.tar import Tar, np_load

def plot_baseline(baseline_archive_filename: str):
    with Tar(baseline_archive_filename) as tar:
        fitnesses_histories = tar.extract('fitness_history.npy', np_load)
    plot_fitness_histories(fitnesses_histories, ['olivedrab', 'honeydew'])


def plot_comparison(baseline_archive_filename, fitness_histories_filename):
    with Tar(baseline_archive_filename) as tar:
        baseline_fitness_histories = tar.extract('fitness_history.npy', np_load)
    fitness_histories = np.load(fitness_histories_filename)
    plot_fitness_histories(-baseline_fitness_histories, colors=['olivedrab', 'honeydew'], show=False)
    plot_fitness_histories(fitness_histories, colors=['orange', 'beige'], show=False)
    plt.show()

def plot_fitness_histories(fitnesses_histories, colors, show=True):
    avg = np.mean(fitnesses_histories, axis=0)
    std = np.std(fitnesses_histories, axis=0)

    plt.plot(np.arange(len(avg)), avg, color=colors[0])
    plt.fill_between(np.arange(len(avg)), avg-std, avg+std, color=colors[1])
    if show:
        plt.show()