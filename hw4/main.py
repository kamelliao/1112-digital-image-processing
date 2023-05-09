from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from mask import all_masks
from digital_halftoning import *
from frequency_domain import *


def dither(image, r, noise):
    dither_matrix = [[0, 2], [3, 1]]
    ditherer = HalftoneDithering(dither_matrix, r=r)
    result = ditherer(image, noise)
    return result


def error_diffusion(image, mask_name):
    mask = all_masks.get(mask_name)
    hed = HalftoneErrorDiffusion(mask)
    result = hed(image)
    return result


def image_sampling(image, dj, dk):
    js, ks = np.meshgrid(
        np.arange(dj//2, image.shape[0], dj),
        np.arange(dk//2, image.shape[1], dk),
        indexing='ij'
    )

    result = image[js, ks]
    return result


def gaussian_highpass(image, d0):
    kernel = highpass_filter(image.shape, d0)
    result = frequency_domain_filtering(image, kernel)
    # plot_filtering(image, kernel).savefig('report_images/vase-compare.png')
    return result.astype(np.uint8)


def remove_pattern(image):
    kernel = vertical_notch_reject_filter(image.shape)
    result = frequency_domain_filtering(image, kernel)
    # plot_filtering(image, kernel).savefig('report_images/dog-compare.png')
    return result.astype(np.uint8)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('command', type=str, choices=['dither', 'error_diffusion', 'image_sample', 'gaussian_highpass', 'remove_pattern'])
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    parser.add_argument('-r', type=int, default=None, help='size of dither matrix')
    parser.add_argument('-n', '--noise', type=float, nargs=2, help='(loc, scale)')
    parser.add_argument('-m', '--mask', type=str, help='mask name for error diffusion')

    parser.add_argument('-dj', type=int, default=25)
    parser.add_argument('-dk', type=int, default=25)
    parser.add_argument('-d0', type=float, default=30.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print(f'{args.command:<12}: {args.input} -> {args.output}')
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if args.command == 'dither':
        result = dither(image, args.r, args.noise)
    elif args.command == 'error_diffusion':
        result = error_diffusion(image, args.mask)
    elif args.command == 'image_sample':
        result = image_sampling(image, args.dj, args.dk)
    elif args.command == 'gaussian_highpass':
        result = gaussian_highpass(image, args.d0)
    elif args.command == 'remove_pattern':
        result = remove_pattern(image)

    cv2.imwrite(args.output, result)
