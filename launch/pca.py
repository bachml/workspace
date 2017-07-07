import numpy as np
import scipy.io as sio
import sys


def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)
    sortArray=sortArray[-1::-1]
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=0)
    newData=dataMat-meanVal
    return newData,meanVal


def pca_n(dataMat, n):
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)

    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(n + 1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    lowDDataMat = newData * n_eigVect
    reconMat = (lowDDataMat * n_eigVect.T) + meanVal
    return lowDDataMat, n_eigVect, meanVal


def pca(dataMat,percentage=0.99):
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    n=percentage2n(eigVals,percentage)
    eigValIndice=np.argsort(eigVals)
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    n_eigVect=eigVects[:,n_eigValIndice]
    lowDDataMat=newData*n_eigVect
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal
    return lowDDataMat,n_eigVect


def pca_base(dataMat, n_eigVect, true_meanVal):
    #newData, meanVal = zeroMean(dataMat)
    newData = dataMat - true_meanVal
    lowDDataMat = newData * n_eigVect
    #reconMat = (lowDDataMat * n_eigVect.T) + meanVal
    return lowDDataMat, n_eigVect


def util_dimRedution(fin, dimension):
    data = np.load(fin)
    reducedData = pca_n(data, dimension)
    np.load(fin + '_reduce', reducedData)


def util_getTransBase(fin, dimension):
    data = np.load(fin)
    (_useless, eigVect, meanVal) = pca_n(data, dimension)
    np.load(fin+'_eigVect', eigVect)
    np.load(fin+'_meanVal', meanVal)


def util_dimReductionFromBase(fin, eigVect, meanVal):
    data = np.load(fin)
    reducedData = pca_base(data, eigVect, meanVal)
    np.load(fin + '_reduce', reducedData)


if __name__ == "__main__":


    #fin = sys.argv[1]
    #dimension = sys.argv[2]

    task_name = sys.argv[1]
    embedding_dim = sys.argv[2]

    # task_path = ../metric_results/{task_...}_intra.npy
    folder = '../metric_results_/'
    intra_feature = np.load( folder + task_name + '_intra.npy')
    extra_feature = np.load( folder + task_name + '_extra.npy')

    feature = np.row_stack((intra_feature, extra_feature))

    (x, new_base, meanVal) = pca_n(feature, int(embedding_dim))


    (pca_intra,xxx) = pca_base(intra_feature,  new_base, meanVal)
    (pca_extra,yyy) = pca_base(extra_feature,  new_base, meanVal)


    np.save(folder + 'pca_' + task_name + '_intra.npy', pca_intra)
    np.save(folder + 'pca_' + task_name + '_extra.npy', pca_extra)


    #np.save(folder + task_name +'_base', new_base)
    #np.save(folder + task_name + 'mean', meanVal)


    sio.savemat(folder + task_name + '_base.mat', {'base': new_base})
    sio.savemat(folder + task_name +  '_mean.mat', {'mean': meanVal})




    '''
    folder = './'
    seeta_intra_feature = np.load(folder + 'true_seeta_intra.npy')
    seeta_extra_feature = np.load(folder + 'true_seeta_extra.npy')

    seeta_feature = np.row_stack((seeta_intra_feature, seeta_extra_feature))
    (x, new_base, meanVal) = pca_n(seeta_feature, dimension)
    print(meanVal)

    print(seeta_feature.shape)
    print(seeta_intra_feature.shape)
    print(new_base.shape)
    (newbase_seeta1240_intra,xxx) = pca_base(seeta_intra_feature, 1240, new_base, meanVal)
    (newbase_seeta1240_extra,yyy) = pca_base(seeta_extra_feature, 1240, new_base, meanVal)

    np.save(folder + 'newbase2_seeta1240_intra.npy', newbase_seeta1240_intra)
    np.save(folder + 'newbase2_seeta1240_extra.npy', newbase_seeta1240_extra)

    np.save(folder + 'pca1240_base.npy', new_base)
    sio.savemat('base1240.mat', {'base': new_base})
    sio.savemat('mean1240.mat', {'base': meanVal})


    folder = 'new_id_test/'
    seeta_intra_feature = np.load(folder + 'seeta_intra.npy')
    seeta_extra_feature = np.load(folder + 'seeta_extra.npy')


    for i in range(1):
        addgap = i*0.05
        precentage = 0.99 + addgap

        (pca_intra,x) = pca_n(seeta_intra_feature, 200)
        (pca_extra,y) = pca_n(seeta_extra_feature, 200)

        np.save(folder + 'seeta_200_intra.npy', pca_intra)
        np.save(folder + 'seeta_200_extra.npy', pca_extra)

        np.save(folder + 'seeta_200_intra_vect.npy', x)
        np.save(folder + 'seeta_200_extra_vect.npy', y)


    for i in range(8):
        morphmat = sio.loadmat('morph_mat/group_' + str(i+1) + '_feat.mat')
        mt = morphmat['feat' + str(i+1)]
        m = np.transpose((mt))
        (low_m, eigvect_m) = pca_n(m,200)
        #np.save('low_morph', low_m)
        #np.save('eidvect', eigvect_m)
        sio.savemat('morph_mat/feat200_' + str(i+1) + '.mat', {'feat'+str(i+1): low_m})

    '''
