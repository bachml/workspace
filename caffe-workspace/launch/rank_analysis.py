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


def rank_analysis(dataMat, percentage):
    newData,meanVal=zeroMean(dataMat)
    covMat=np.cov(newData,rowvar=0)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))

    n=percentage2n(eigVals,percentage)
    eigValIndice=np.argsort(eigVals)
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    n_eigVect=eigVects[:,n_eigValIndice]
    lowDDataMat=newData*n_eigVect
    reconMat=(lowDDataMat*n_eigVect.T)+meanVal
    #return lowDDataMat,n_eigVect
    print('precentage: ' + str(percentage))
    print('shape: ' )
    print(lowDDataMat.shape)




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
    data = np.load(sys.argv[1])
    data = np.reshape(data, (512,20))
    data = np.transpose(data)
    print(data.shape)
    ratio = float(sys.argv[2])
    rank_analysis(data, ratio)


