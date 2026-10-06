"""Plot exact post-update losses and gradient norms from a training checkpoint."""
from pathlib import Path
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from veros_control.core.checkpoint import load_checkpoint


def plot_training(checkpoint,output):
    c=load_checkpoint(checkpoint);history=c['train_history'];out=Path(output);out.mkdir(parents=True,exist_ok=True)
    fields=('update','loss','post_update_loss','learning_rate','gradient_norm','seconds')
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(history)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot([x['update'] for x in history],[x['post_update_loss'] for x in history])
    axes[0].set_ylabel('Normalized sampled loss' if c['config']['loss']['balance'] else 'Sampled loss')
    axes[1].plot([x['update'] for x in history],[x['gradient_norm'] for x in history]);axes[1].set_ylabel('Gradient norm');axes[1].set_yscale('log')
    for ax in axes:ax.set_xlabel('Update');ax.grid(alpha=.2)
    fig.savefig(out/'training.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('output');a=p.parse_args();plot_training(a.checkpoint,a.output)


if __name__=='__main__':main()
