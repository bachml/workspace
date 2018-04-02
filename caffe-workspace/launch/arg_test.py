import sys
import argparse


def parse_args():

    parser = argparse.ArgumentParser(description='Generate ResNet Prototxt')
    parser.add_argument('--cfg', dest='cfg_file', default=None, type=str,
                        help='Definition of the network')
    parser.add_argument('-t', '--type', dest='type', default='fasterrcnn', type=str,
                        help='Network for fasterrcnn, fastrcnn, or classification')


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = parse_args()
    
    x = arg.type
    print(x)
