from argparse import ArgumentParser
from warm_up import *
from image_enhancement import *
from noise_removal import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('operation', type=str)
    parser.add_argument('-g', '--gray-scale', action='store_true')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    transformations = {
        'grayscale': grayscale,
        'vflip': flip,
        'dec_brightness': dec_brightness,
        'inc_brightness': inc_brightness,
        'global_hist_equalize': global_hist_equalize,
        'local_hist_equalize': local_hist_equalize,
        'transfer_func': transfer_func,
        'spatial_filter': spatial_filter,
        'pmed_filter': pmed_filter
    }

    if args.operation == 'psnr':
        img1 = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(args.output, cv2.IMREAD_GRAYSCALE)
        psnr_value = psnr(img1, img2)
        print(f'PSNR value = {psnr_value}')
    else:
        print(f'{args.operation:<20}: {args.input} -> {args.output}')
        if args.gray_scale:
            image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(args.input)
        result = transformations.get(args.operation)(image)
        cv2.imwrite(args.output, result)
