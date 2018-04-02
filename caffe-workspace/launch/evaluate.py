import sys
import numpy as np
from show_accuracy import *
import argparse

def parse_args():


    parser = argparse.ArgumentParser(description='Evaluate Verification Performance')

    parser.add_argument('--tn', dest='task_name', default='None', type=str,
                        help='task_name')

    parser.add_argument('--n', dest='num_pairs', default=3000, type=int,
			help='number of verification pairs')


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



def metric_BrayCurtis(x1, x2):
    x1.shape = (-1,1)
    x2.shape = (-1,1)

    temp = np.linalg.norm(x1 - x2, 1) / np.linalg.norm(x1 + x2, 1)
    ratio = 120 - 80*temp
    return float(ratio)

def metric_L2(x1,x2):
    x1.shape = (-1,1)
    x2.shape = (-1,1)
    #x1 = x1 / np.linalg.norm(x1)
    #x2 = x2 / np.linalg.norm(x2)
    z = np.linalg.norm(x1 - x2, 2)
    ratio = -1 * z
    return float(ratio)

def metric_cosine100(x1, x2):
    x1.shape = (-1,1)
    x2.shape = (-1,1)

    ratio =np.dot(np.transpose(x1), x2) / ( np.sqrt(np.dot(np.transpose(x1),x1)) * np.sqrt(np.dot(np.transpose(x2),x2)) )
    ratio = (ratio + 1)*50

    return float(ratio)


def metric_cosine(x1, x2):
    x1.shape = (-1,1)
    x2.shape = (-1,1)

    ratio =np.dot(np.transpose(x1), x2) / ( np.sqrt(np.dot(np.transpose(x1),x1)) * np.sqrt(np.dot(np.transpose(x2),x2)) )

    return float(ratio)

if __name__ == "__main__":


    basepath = '/home/zeng/workspace/metric_results_/'
    name = "facenet"
    folder = "./"

    args = parse_args()
    task_name = args.task_name
    intra_feature = np.load(basepath + args.task_name + "_intra.npy")
    extra_feature = np.load(basepath + args.task_name + "_extra.npy")
    intra_dist = np.empty(shape=[0, 1])
    extra_dist = np.empty(shape=[0, 1])



    for i in range(args.num_pairs * 2):
        if i%2 == 1:
            continue
        intraA = intra_feature[ [i], :]
        intraB = intra_feature[ [i+1], :]
        extraA = extra_feature[ [i], :]
        extraB = extra_feature[ [i+1], :]
        #sim = metric_BrayCurtis(A, B)
        sim = metric_cosine100(intraA, intraB)
        intra_dist = np.vstack((intra_dist, sim))
        sim = metric_cosine100(extraA, extraB)
        extra_dist = np.vstack((extra_dist, sim))

    show_accuracy(intra_dist, extra_dist)
    np.save(basepath + args.task_name + "_dist_intra.npy", intra_dist)
    np.save(basepath + args.task_name + "_dist_extra.npy", extra_dist)

