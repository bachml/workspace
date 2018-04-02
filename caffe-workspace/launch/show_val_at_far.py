import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.externals import joblib

def report_format(report):
    report = report.split()
    result = np.zeros((3,3))
    result[0][0] = report[5]
    result[0][1] = report[6]
    result[0][2] = report[7]

    result[1][0] = report[10]
    result[1][1] = report[11]
    result[1][2] = report[12]

    result[2][0] = report[17]
    result[2][1] = report[18]
    result[2][2] = report[19]

    return result


def calculate_val_far(threshold, predict_issame, actual_issame):
    #predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far



def show_val_at_far(dist_intra, dist_extra, target_far):
    print(type(target_far))
    dist_all = np.append(dist_intra, dist_extra)
    label = np.append(np.repeat(1, len(dist_intra)), np.repeat(0, len(dist_extra)))

    dist = dist_all
    y    = label

    t_s = np.min(dist_all)
    t_e = np.max(dist_all)
    t_step = (t_e - t_s) / 1000

    optimized_gap = np.inf
    result_far = 0
    result_val = 0
    result_threshold = 0

    while(t_s < t_e):
        pre = dist >= t_s
        y = (y==1)
        (val, far) = calculate_val_far(t_s, pre, y)
        #print(type(far))
        gap = np.fabs(far - target_far)
        if gap < optimized_gap:
            optimized_gap = gap
            result_far = far
            result_val = val
            result_threshold = t_s
        t_s += t_step

    print "far: ", result_far
    print "val: ", result_val
    print "threshold: ", result_threshold
