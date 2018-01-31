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
    #feature = feature *10

    (x, new_base, meanVal) = pca_n(feature, int(embedding_dim))


    (pca_intra,xxx) = pca_base(intra_feature,  new_base, meanVal)
    (pca_extra,yyy) = pca_base(extra_feature,  new_base, meanVal)

    pca_intra = np.real(pca_intra)
    pca_extra = np.real(pca_extra)


    np.save(folder + 'pca_' + task_name + '_intra.npy', pca_intra)
    np.save(folder + 'pca_' + task_name + '_extra.npy', pca_extra)


    #np.save(folder + task_name +'_base', new_base)
    #np.save(folder + task_name + 'mean', meanVal)

    np.save('base', new_base)
    np.save('mean', meanVal)


    sio.savemat(folder + task_name + '_base.mat', {'base': new_base})
    sio.savemat(folder + task_name +  '_mean.mat', {'mean': meanVal})


