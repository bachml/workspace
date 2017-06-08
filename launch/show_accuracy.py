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

def show_accuracy(dist_intra, dist_extra):

    dist_all = np.append(dist_intra, dist_extra)
    label = np.append(np.repeat(1, len(dist_intra)), np.repeat(0, len(dist_extra)))

    dist = dist_all
    y    = label

    t_s = np.min(dist_all)
    t_e = np.max(dist_all)
    t_step = (t_e - t_s) / 1000

    optimized_acc = -np.inf
    optimized_threshold = -np.inf

    while(t_s < t_e):
        pre = dist >= t_s
        y = (y==1)
        report = metrics.classification_report(y_true=y, y_pred=pre)
        acc = accuracy_score(y_true=y, y_pred=pre)
        #print "accurate: "
        #print  acc
        #print "threshold: ", t_s
        #print report
        report_result = report_format(report)

        t_s += t_step

	if optimized_acc < acc :
	    optimized_acc = acc
	    optimized_threshold = t_s
    print "accurate: ", optimized_acc 
    print "threshold: ", optimized_threshold



if __name__ == "__main__":

    folder_path = "metric_results/"
    #show_accracy(folder_path, 'xx_dist_intra.npy', 'xx_dist_extra.npy')

   









