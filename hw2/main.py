from argparse import ArgumentParser
import cv2
from edge_detection import *
from hough_transform import *
from geometrical_modification import *

def build_parser():
    ioparser = ArgumentParser(add_help=False)
    ioparser.add_argument('-i', '--input', type=str, required=True)
    ioparser.add_argument('-o', '--output', type=str, required=True)

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title='command', dest='command', required=True)
    
    sobel_parser = subparsers.add_parser('sobel', parents=[ioparser])
    sobel_parser.add_argument('-t', '--threshold', type=int, default=None)
    
    canny_parser = subparsers.add_parser('canny', parents=[ioparser])
    canny_parser.add_argument('-tc', '--threshold_cand', type=int, default=40)
    canny_parser.add_argument('-te', '--threshold_edge', type=int, default=75)
    canny_parser.add_argument('-n', '--neighbor', default='eight', choices=['four', 'eight'])

    log_parser = subparsers.add_parser('log', parents=[ioparser])
    log_parser.add_argument('-g', '--grad', action='store_true')
    log_parser.add_argument('-t', '--threshold', type=float, default=2)
    log_parser.add_argument('-n', '--neighbor', type=str, default='eight_sep', choices=['four', 'eight_sep', 'eight_nonsep'])

    edge_crisp_parser = subparsers.add_parser('edge_crisp', parents=[ioparser])
    edge_crisp_parser.add_argument('-l', type=int, default=5)
    edge_crisp_parser.add_argument('-c', type=float, default=0.7)

    hough_parser = subparsers.add_parser('hough', parents=[ioparser])

    borzoi_parser = subparsers.add_parser('borzoi', parents=[ioparser])
    popdog_parser = subparsers.add_parser('popdog', parents=[ioparser])

    return parser

def cmd_sobel_edge_detection(image, args):
    return sobel_edge_detection(image, threshold=args.threshold)

def cmd_canny_edge_detection(image, args):
    canny_edge_detectection = CannyEdgeDetector()
    return canny_edge_detectection(image, th_cand=args.threshold_cand, th_edge=args.threshold_edge, neighbor=args.neighbor)

def cmd_edge_crisp(image, args):
    return edge_crispening(image, L=args.l, c=args.c)

def cmd_laplacian_of_gaussian(image, args):
    return laplacian_of_gaussian(image, grad=args.grad, neighbor=args.neighbor, threshold=args.threshold)

def cmd_hough(image, args):
    hough = HoughSpace()
    hough.transform(image)
    fig = hough.plot_hough_space()
    return fig

def cmd_borzoi(image, args):
    # segment the borzoi
    components = segmentation(image)
    components = sorted(components, key=lambda x: x.shape[0], reverse=True)
    dog_coord = components[0]
    result = np.full(image.shape, 255)
    result[dog_coord.T[0], dog_coord.T[1]] = image[dog_coord.T[0], dog_coord.T[1]]

    # perform transformation on borzoi
    trans = [
        scaling_bt(0.9, 1.8),
        translation_bt(400, 100),
        rotation_bt(np.deg2rad(-40)),
        translation_bt(-450, -320),
    ]
    result = linear_transformation(result, trans)

    # paste the rest elements back
    for comp in components[1:]:
        result[comp.T[0], comp.T[1]] = image[comp.T[0], comp.T[1]]
    return result

def cmd_popdog(image, *args):
    h, w = image.shape
    center = np.array([[w//2-12, h//2-12]]).astype(int)
    return popcat(image, center)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        'sobel': cmd_sobel_edge_detection,
        'canny': cmd_canny_edge_detection,
        'edge_crisp': cmd_edge_crisp,
        'log': cmd_laplacian_of_gaussian,
        'hough': cmd_hough,
        'borzoi': cmd_borzoi,
        'popdog': cmd_popdog
    }

    print(f'{args.command:<12}: {args.input} -> {args.output}')
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    result = commands.get(args.command)(image, args)
    if args.command == 'hough':
        result.savefig(args.output)
    else:
        cv2.imwrite(args.output, result)
