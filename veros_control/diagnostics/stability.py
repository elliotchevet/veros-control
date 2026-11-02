"""Daily envelopes of hourly stability checks along paired trajectories."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt


def plot_stability(directory):
    root = Path(directory)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for label in ('baseline', 'trained'):
        with (root/f'{label}_stability.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        years = [float(row['step'])/8640 for row in rows]
        for ax, key in zip(axes.flat, ('max_cfl', 'min_salinity', 'min_temperature', 'max_temperature')):
            ax.plot(years, [float(row[key]) for row in rows], label=label)
            ax.set(title=key, xlabel='Model years'); ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Daily envelopes of hourly checks')
    for extension in ('png', 'pdf'):
        fig.savefig(root/f'stability.{extension}', dpi=150)
    plt.close(fig)
