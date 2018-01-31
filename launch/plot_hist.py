import sys
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt


def fitData(filepath):
    data = np.load(filepath)
    cnts, bins = np.histogram(data, bins=100, normed=True)
    bins = (bins[:-1] + bins[1:]) / 2
    y = np.polyfit(bins, cnts, 2)
    f_liner = np.polyval(y, bins)
    #return bins,f_liner
    return bins,cnts

def fitData_cos(filepath):
    data = (np.load(filepath) + 1) * 50
    cnts, bins = np.histogram(data, bins=100, normed=True)
    bins = (bins[:-1] + bins[1:]) / 2
    y = np.polyfit(bins, cnts, 2)
    f_liner = np.polyval(y, bins)
    #return bins,f_liner
    return bins,cnts

if __name__ == '__main__':



    '''(x1, y1) = fitData_cos('id_test/aifr_dist_intra.npy')
    plt.plot(x1, y1, label=u'x', color='g', linestyle='-', marker='')
    (x2, y2) = fitData_cos('id_test/aifr_dist_extra_5000.npy')
    plt.plot(x2, y2, label=u'x', color='g', linestyle='-', marker='')

    (x3, y3) = fitData('id_test/ts_dist_intra.npy')
    plt.plot(x3, y3, label=u'x', color='r', linestyle='-', marker='')
    (x4, y4) = fitData('id_test/ts_dist_extra_5000.npy')
    plt.plot(x4, y4, label=u'x', color='r', linestyle='-', marker='')

    data = np.load('id_test/ts_dist_intra.npy')
    data = data[0:3000]
    sns.kdeplot(data, shade=True,color='r', label = 'taisau')
    data_y = np.load('id_test/ts_dist_extra_5000.npy')
    data_y = data_y[0:3000]
    sns.kdeplot(data_y, shade=True,color='r')

    data2 = (np.load('id_test/aifr_dist_intra.npy') + 1 ) * 50
    data2 = data2[0:3000]
    data2 = data2.reshape(3000)
    sns.kdeplot(data2, shade=True, color='g', label = 'cross-age deepid')
    data2_y = (np.load('id_test/aifr_dist_extra_5000.npy') + 1 ) * 50
    data2_y = data2_y[0:3000]
    data2_y = data2_y.reshape(3000)
    sns.kdeplot(data2_y, shade=True, color='g')

    data_224x192 = (np.load('new_id_test/id_elu_notreal_224x192_dist_intra.npy') + 1 ) * 50
    data_224x192 = data_224x192[0:3000]
    data_224x192 = data_224x192.reshape(3000)
    sns.kdeplot(data_224x192, shade=True, color='w', label = 'id_elu_224x192')
    data_224x192_y = (np.load('new_id_test/id_elu_notreal_224x192_dist_extra.npy') + 1 ) * 50
    data_224x192_y = data_224x192_y[0:3000]
    data_224x192_y = data_224x192_y.reshape(3000)
    sns.kdeplot(data_224x192_y, shade=True, color='w')

    '''
    '''
    data3 = (np.load('new_id_test/true_seeta_Cosine_dist_intra.npy') )
    data3 = ((data3)  *2) * 50
    data3 = data3.reshape(3000)
    sns.kdeplot(data3, shade=True, color='g', label = 'GFace6')
    data3_y = (np.load('new_id_test/true_seeta_Cosine_dist_extra.npy') )
    data3_y = ((data3_y)  *2) * 50
    data3_y = data3_y.reshape(3000)
    sns.kdeplot(data3_y, shade=True, color='g')


    data4 = (np.load('new_id_test/newbase2_seeta1240_BrayCurtis_dist_intra.npy') *-1 )+1
    data4 = ((data4 *2 + 1)  *50 )
    data4 = data4.reshape(3000)
    sns.kdeplot(data4, shade=True, color='b', label = 'GFace6.2')
    data4_y = (np.load('new_id_test/newbase2_seeta1240_BrayCurtis_dist_extra.npy') *-1 )  +1
    data4_y = ((data4_y*2 +1 )  *50)
    data4_y = data4_y.reshape(3000)
    sns.kdeplot(data4_y, shade=True, color='b')

    data5 = (np.load('new_id_test/newbase2_seeta1240_Cosine_dist_intra.npy') )
    data5 = ((data5 +1) ) * 50
    data5 = data5.reshape(3000)
    sns.kdeplot(data5, shade=True, color='r', label = 'GFace6.1')
    data5_y = (np.load('new_id_test/newbase2_seeta1240_Cosine_dist_extra.npy') )
    data5_y = ((data5_y +1) ) * 50
    data5_y = data5_y.reshape(3000)
    sns.kdeplot(data5_y, shade=True, color='r')
    #sns.rugplot(data, color="#CF3512")
    #data_y = np.load('id_test/aifr_dist_extra_5000.npy')
    #sns.kdeplot(data_y, shade=True)


    data = (np.load('GFace7_dist_intra.npy') )
    #data = ((data)*0.8  +1.2)*100
    data = data.reshape(3000)
    sns.kdeplot(data, shade=True, color='r', label = 'GFace7')
    data_y = (np.load('GFace7_dist_extra.npy') )
    #data_y = ((data_y)*0.8 + 1.2)*100
    data_y = data_y.reshape(3000)
    sns.kdeplot(data_y, shade=True, color='r')
    '''
    task_name = sys.argv[1]
    path = '../metric_results_/'
    intra_data = np.load(path + task_name + '_dist_intra.npy')
    intra_data = intra_data.reshape(3000)
    sns.kdeplot(intra_data, shade=True, color='r', label = task_name)
    extra_data = np.load(path + task_name + '_dist_extra.npy')
    extra_data = extra_data.reshape(3000)
    sns.kdeplot(extra_data, shade=True, color='r')


    plt.show()

