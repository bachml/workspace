import numpy as np

name = "facenet"
folder = "./"

intra = np.load(folder + name + "_intra.npy")
exter = np.load(folder + name + "_extra.npy")


def Verify_regBrayCurtis(x1, x2):
    x1.shape = (-1,1)
    x2.shape = (-1,1)

    x1 = x1 / np.linalg.norm(x1)
    x2 = x2 / np.linalg.norm(x2)

    ratio = np.linalg.norm(x1 - x2, 1) / np.linalg.norm(x1 + x2, 1)
    return float(ratio)


def Verify_BrayCurtis(x1, x2):
    x1.shape = (-1,1)
    x2.shape = (-1,1)

    temp = np.linalg.norm(x1 - x2, 1) / np.linalg.norm(x1 + x2, 1)
    ratio = 120 - 80*temp
    return float(ratio)

def Verify_L2(x1,x2):
    x1.shape = (-1,1)
    x2.shape = (-1,1)

    #x1 = x1 / np.linalg.norm(x1)
    #x2 = x2 / np.linalg.norm(x2)

    z = np.linalg.norm(x1 - x2, 2)
    ratio = -1 * z

    return float(ratio)

def Verify(x1, x2):
    x1.shape = (-1,1)
    x2.shape = (-1,1)

    ratio =np.dot(np.transpose(x1), x2) / ( np.sqrt(np.dot(np.transpose(x1),x1)) * np.sqrt(np.dot(np.transpose(x2),x2)) )

    return float(ratio)

if __name__ == "__main__":

    nan = np.empty(shape=[0, 1])

    for i in range(6000):
        if i%2 == 1:
            continue
        A = intra[ [i], :]
        B = intra[ [i+1], :]

        #sim = Verify_BrayCurtis(A, B)
        sim = Verify(A, B)
        nan = np.vstack((nan, sim))

    print nan.shape
    np.save(folder + name + "_dist_intra.npy", nan)

    nan = np.empty(shape=[0, 1])
    for i in range(6000):
        if i%2 == 1:
            continue
        A = exter[ [i], :]
        B = exter[ [i+1], :]

        #sim = Verify_BrayCurtis(A, B)
        sim = Verify(A, B)
        nan = np.vstack((nan, sim))

    print nan.shape
    np.save(folder + name + "_dist_extra.npy", nan)
