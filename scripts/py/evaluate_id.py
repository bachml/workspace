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


def show_roc(folder, intradist_path, extradist_path, plot_name):
    dist_intra = np.load(folder + "/" + intradist_path)
    dist_extra = np.load(folder + "/" + extradist_path)

    dist_all = np.append(dist_intra, dist_extra)
    label = np.append(np.repeat(1, len(dist_intra)), np.repeat(0, len(dist_extra)))

    fpr, tpr, _ = roc_curve(label, dist_all, pos_label=1)
    plt.plot(fpr, tpr, label=plot_name)


def show_accracy(folder, intradist_path, extradist_path, t_s, t_e, t_step):
    dist_intra = np.load(folder + "/" + intradist_path)
    dist_extra = np.load(folder + "/" + extradist_path)

    dist_all = np.append(dist_intra, dist_extra)
    label = np.append(np.repeat(1, len(dist_intra)), np.repeat(0, len(dist_extra)))

    dist = dist_all
    y    = label


    while(t_s < t_e):
        pre = dist >= t_s
        y = (y==1)
        report = metrics.classification_report(y_true=y, y_pred=pre)
        acc = accuracy_score(y_true=y, y_pred=pre)
        print "accurate: "
        print  acc
        print "threshold: ", t_s
        print report
        report_result = report_format(report)

        t_s += t_step



if __name__ == "__main__":

    folder_path = "."
    #show_roc(folder_path, 'sface_dist_intra.npy', 'sface_dist_extra.npy', 'seetaface/256x256')
    #show_roc(folder_path, 'seeta_dist_intra.npy', 'seeta_dist_extra.npy', 'seetaface/seeta')

    #show_roc(folder_path, 'relu_dist_intra.npy', 'relu_dist_extra.npy', 'relu')
    #show_roc(folder_path, 'id_wen_dist_intra.npy', 'id_wen_dist_extra.npy', 'GFace7')


    #show_roc(folder_path, 'seeta_0.7_dist_intra.npy', 'seeta_0.7_dist_extra.npy', 'seetaface/seeta0.7')
    #show_roc(folder_path, 'seeta_0.75_dist_intra.npy', 'seeta_0.75_dist_extra.npy', 'seetaface/seeta0.75')
    
    #show_roc(folder_path, 'GFace7_dist_intra.npy', 'GFace7_dist_extra.npy','GFace7')
    #show_roc(folder_path, 'pca200_LightenedCNNregL2_dist_intra.npy', 'pca200_LightenedCNNregL2_dist_extra.npy','GFace7regL2BC')
    #show_roc(folder_path, 'pca200_LightenedCNNreg_dist_intra.npy', 'pca200_LightenedCNNreg_dist_extra.npy','GFace7regBC')
    #show_roc(folder_path, 'GFace6.2_dist_intra.npy', 'GFace6.2_dist_extra.npy','GFace6.2')
    #show_roc(folder_path, 'GFace6_dist_intra.npy', 'GFace6_dist_extra.npy','GFace6')
    #show_roc(folder_path, 'LightenedCNN_lmdb_intra.npy', 'LightenedCNN_lmdb_extra.npy','new')
    
    #show_roc(folder_path, 'aifr_5w_seeta_dist_intra.npy', 'aifr_5w_seeta_dist_extra.npy', 'seetaface/seeta_5w_aifr_200')
    #show_roc(folder_path, 'aifr_x_seeta_dist_intra.npy', 'aifr_x_seeta_dist_extra.npy','seetaface/seeta_aifr')
    #show_roc(folder_path, 'aifr_52048_seeta_dist_intra.npy', 'aifr_52048_seeta_dist_extra.npy', 'seetaface/seeta_5aifr_2048')
    #show_roc(folder_path, 'newbase_seeta200_dist_intra.npy', 'newbase_seeta200_dist_extra.npy', 'seetaface/newbase_seeta')

    #show_roc(folder_path, 'seeta_200_dist_intra.npy', 'seeta_200_dist_extra.npy', 'seetaface/seeta200')
    #show_roc(folder_path, 'seeta_0.8_dist_intra.npy', 'seeta_0.8_dist_extra.npy', 'seetaface/seeta200 + cross-age')
    #show_roc(folder_path, 'seeta_0.8_dist_intra.npy', 'seeta_0.8_dist_extra.npy', 'seetaface/seeta0.8')
    #show_roc(folder_path, 'seeta_0.85_dist_intra.npy', 'seeta_0.85_dist_extra.npy', 'seetaface/seeta0.85')
    #show_roc(folder_path, 'seeta_0.9_dist_intra.npy', 'seeta_0.9_dist_extra.npy', 'seetaface/seeta0.9')
    #show_roc(folder_path, 'seeta_0.95_dist_intra.npy', 'seeta_0.95_dist_extra.npy', 'seetaface/seeta0.95')
    #show_roc(folder_path, 'seeta_0.99_dist_intra.npy', 'seeta_0.99_dist_extra.npy', 'seetaface/seeta0.99')

    #show_roc(folder_path, 'elu_84dist_intra.npy', 'elu_dist_extra.npy', 'deepid/112x96')
    #show_roc(folder_path, 'id_elu_224x192_dist_intra.npy', 'id_elu_224x192_dist_extra.npy', 'deepid + elu224x192')
    #show_roc(folder_path, 'id_elu_112x96_dist_intra.npy', 'id_elu_112x96_dist_extra.npy','deepid/112x96')
    #show_roc(folder_path, 'id_elu_notreal_224x192_dist_intra.npy', 'id_elu_notreal_224x192_dist_extra.npy', 'deepid-tuned-A/224x192')
    #show_roc(folder_path, 'id_elu_conv_224x192_dist_intra.npy', 'id_elu_conv_224x192_dist_extra.npy', 'deepid-tuned-B/224x192')
    #show_roc(folder_path, 'id_elu_real_224x192_dist_intra.npy', 'id_elu_real_224x192_dist_extra.npy', 'deepid-tuned-C/224x192')
    #show_roc(folder_path, 'elu_84x72_dist_intra.npy', 'elu_84x72_dist_extra.npy', 'deepid/84x72')
    #show_roc(folder_path, 'elu_98x84_dist_intra.npy', 'elu_98x84_dist_extra.npy', 'deepid + elu98x84')
    #show_roc(folder_path, 'elu_70x60_dist_intra.npy', 'elu_70x60_dist_extra.npy', 'deepid + elu70x60')
    #show_roc(folder_path, 'LeNetlike_deepid_dist_intra.npy', 'LeNetlike_deepid_dist_extra.npy', 'LeNetlike_deepi')
    #show_roc(folder_path, 'wen_cvpr_50x50_dist_intra.npy', 'wen_cvpr_50x50_dist_extra.npy', 'wen')
    #show_roc(folder_path, 'elu0.8_dist_intra.npy', 'elu0.8_dist_extra.npy', 'deepidx0.8 + elu + jb')
    #show_roc(folder_path, 'elu0.8jb_dist_intra.npy', 'elu0.8jb_dist_extra.npy', 'deepidx0.8 + elu + jb')
    #show_roc(folder_path, 'aifr_dist_intra.npy', 'aifr_dist_extra.npy', 'deepid + aifr')

    #show_roc(folder_path, 'elu_84x72_dist_intra.npy', 'elu_84x72_dist_extra.npy', 'deepid_84x72_aligned')
    #show_roc(folder_path, 'elu_dist_intra.npy', 'elu_dist_extra.npy', 'deepid_50x50')
    #show_roc(folder_path, 'ts_dist_intra.npy', 'ts_dist_extra.npy', 'taisau_50x50')


    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    #plt.title('ROC curve (test on ID verification)')
    #plt.legend(loc='best')

    #show_accracy(folder_path, 'pca200_LightenedCNN_g7dist_intra.npy', 'pca200_LightenedCNN_g7dist_extra.npy', 45, 55, 0.1)
    show_accracy(folder_path, 'pca200_LightenedCNN_lmdb_dist_intra.npy', 'pca200_LightenedCNN_lmdb_dist_extra.npy',48,52,0.05)

    #plt.show()








