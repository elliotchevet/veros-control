"""Learning curves on the changing training batch and fixed checkpoint panel."""
from pathlib import Path
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from veros_control.core.checkpoint import load_checkpoint


def plot_training(checkpoint, output):
    saved = load_checkpoint(checkpoint)
    history = saved['train_history']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ('update', 'loss', 'post_update_loss', 'reference_loss', 'learning_rate',
              'next_lr', 'gradient_norm', 'seconds')
    with (output/'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(history)
    updates = [row['update'] for row in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, term in zip(axes[0], (None, 'temp', 'salt')):
        for key, label, style in [('post_update_loss', 'Training batch', '-'),
                                  ('reference_loss', 'Fixed reference panel', '--')]:
            if term:
                values = [row.get(key+'_terms', {}).get(term, float('nan')) /
                          saved['loss_term_denominators'][term] for row in history]
            else:
                values = [row.get(key, float('nan')) for row in history]
            ax.plot(updates, values, style, label=label)
        ax.set_title('Combined loss' if term is None else term)
        ax.legend()
    axes[1, 0].plot(updates, [row['learning_rate'] for row in history], label='Applied LR')
    axes[1, 0].plot(updates, [row.get('next_lr', row['learning_rate']) for row in history], '--', label='Next LR')
    axes[1, 0].set_yscale('log'); axes[1, 0].legend()
    axes[1, 1].plot(updates, [row['gradient_norm'] for row in history])
    axes[1, 1].set(title='Gradient norm', yscale='log')
    axes[1, 2].plot(updates, [row['seconds'] for row in history])
    axes[1, 2].set_title('Seconds per update')
    for ax in axes.flat:
        ax.set_xlabel('Update'); ax.grid(alpha=.2)
    fig.suptitle(f"Best fixed-panel update: {saved['best_update']}")
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'training.{extension}', dpi=150)
    plt.close(fig)
    return output/'training.png'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint'); parser.add_argument('output')
    args = parser.parse_args()
    print(plot_training(args.checkpoint, args.output))


if __name__ == '__main__':
    main()
