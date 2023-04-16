from argparse import ArgumentParser
import cv2
from morphological_processing import *
from texture_analysis import *

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('command', type=str)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    # for texture analysis
    parser.add_argument('-p', '--pool', type=str, default='mean')
    parser.add_argument('-r', '--r', type=int, default=17)
    parser.add_argument('--pos', action='store_true')
    parser.add_argument('--n_clusters', type=int, default=4)
    return parser

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        'boundary': boundary_extraction,
        'holefill': hole_filling,
        'count': object_counting,
        'open': open_op,
        'close': close_op,
        'texture': texture_segmentation,
    }

    print(f'{args.command:<12}: {args.input} -> {args.output}')
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    if args.command == 'count':
        n_object, fig = commands.get(args.command)(image)
        print(f'Number of objects = {n_object}')
        fig.savefig(args.output, dpi=1000, bbox_inches='tight')
    elif args.command == 'texture':
        result, labels = commands.get(args.command)(image, args)
        result.savefig(args.output, bbox_inches='tight')
        np.save('labels.npy', labels)
    else:
        result = commands.get(args.command)(image)
        cv2.imwrite(args.output, result)
