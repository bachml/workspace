import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.externals import joblib



def show_roc(folder, intradist_path, extradist_path, plot_name):
    dist_intra = np.load(folder + "/" + intradist_path)
    dist_extra = np.load(folder + "/" + extradist_path)
    dist_all = np.append(dist_intra, dist_extra)
    label = np.append(np.repeat(1, len(dist_intra)), np.repeat(0, len(dist_extra)))

    fpr, tpr, _ = roc_curve(label, dist_all, pos_label=1)
    plt.plot(fpr, tpr, label=plot_name)




if __name__ == "__main__":
   
    folder_path = './' 
    show_roc(folder_path, 'GFace7_dist_intra.npy', 'GFace7_dist_extra.npy','GFace7')



    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (test on ID verification)')
    plt.legend(loc='best')
    #plt.show()

    plt.savefig('result.jpg')








