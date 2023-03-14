from argparse import ArgumentParser
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_transfer_function(transfer_func: np.ndarray):
    fig, ax = plt.subplots()
    ax.plot(range(0, 256), transfer_func)
    ax.axline((0, 0), slope=1, color='gray', linestyle='dashed', alpha=.5)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    return fig

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    fig, axs = plt.subplots(ncols=len(args.input), sharex=True, sharey=True, figsize=(5*len(args.input), 5))
    if len(args.input) == 1:
        img = cv2.imread(args.input[0], cv2.IMREAD_GRAYSCALE)
        axs.hist(img.ravel(), 256, [0, 256])
        axs.set_title(args.input[0])
    else:
        for i, filename in enumerate(args.input):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            axs[i].hist(img.ravel(), 256, [0, 256])
            axs[i].set_title(filename)

        for ax in axs.flat:
            ax.label_outer()

    fig.tight_layout(pad=2)
    fig.savefig(args.output)
