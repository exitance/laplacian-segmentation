import os
import numpy as np
from numpy.linalg import eig

eps = 1e-8

def unnormalized_laplacian(adjMatrix):
    R = np.sum(adjMatrix, axis=1)
    degreeMatrix = np.diag(R)
    return degreeMatrix - adjMatrix

def normalized_laplacian(adjMatrix):
    R = np.sum(adjMatrix, axis=1)
    msqrtR = 1 / np.sqrt(R)
    msqrtD = np.diag(msqrtR)
    I = np.eye(adjMatrix.shape[0])
    return I - np.matmul(msqrtD, np.matmul(adjMatrix, msqrtD))

def ascending_eigenValue(M, dim):
    eigv, _ = np.linalg.eig(M)
    eigv = np.sort(eigv)
    v = np.zeros(dim)
    v[0:eigv.size] = eigv
    v[eigv.size:dim] = eigv[eigv.size-1]
    return v

def create_direcotry(path):
    folder = os.path.exists(path)

    if not folder:
        # recursively
        os.makedirs(path)
    else:
        print("folder %s exists" % (path))

def get_num(filePath):
    f = open(filePath)
    verticesNum = 0
    for line in f:
        if line[0] == "v":
            verticesNum += 1
        if line[0] == "f":
            break
    return verticesNum

def get_adjacent_matrix(filePath):
    f = open(filePath)
    verticesNum = 0
    fst = True
    adjMatrix = []
    for line in f:
        if line[0] == "v":
            verticesNum += 1
        if line[0] == "f":
            if fst:
                adjMatrix = np.zeros((verticesNum, verticesNum))
                fst = False
            face = line.split(" ")
            a = int(face[1]) - 1
            b = int(face[2]) - 1
            c = int(face[3]) - 1
            # print("{0} {1} {2}".format(a, b, c))
            adjMatrix[a][b] = adjMatrix[b][a] = 1
            adjMatrix[a][c] = adjMatrix[c][a] = 1
            adjMatrix[b][c] = adjMatrix[c][b] = 1
    return adjMatrix

def get_segnum(filePath):
    f = open(filePath)
    seglable = []
    segnum = 0
    for line in f:
        x = int(line)
        if x not in seglable:
            segnum += 1
            seglable.append(x)
    return segnum

def get_max_dim(sourceDir):
    path = sourceDir + "/train"
    files = os.listdir(path)
    dim = 0
    for file in files:
        filePath = path + "/" + file
        if not os.path.isdir(filePath):
            num = get_num(path + "/" + file)
            dim = max(num, dim)
    path = sourceDir + "/test"
    files = os.listdir(path)
    for file in files:
        filePath = path + "/" + file
        if not os.path.isdir(filePath):
            num = get_num(path + "/" + file)
            dim = max(num, dim)
    print("maximum dimension: {0}".format(dim))
    return dim

def append_data_label(sourceDir, objectDir, workDir, labelList, normalized, dim):
    path = sourceDir + "/" + workDir
    files = os.listdir(path)
    for file in files:
        filePath = path + "/" + file
        if not os.path.isdir(filePath):
            adjMatrix = get_adjacent_matrix(path + "/" + file)
            if normalized == False:
                laplacian = unnormalized_laplacian(adjMatrix)
            else:
                laplacian = normalized_laplacian(adjMatrix)
            eigenValues = ascending_eigenValue(laplacian, dim)
            lowerBound = 0-eps
            upperBound = 2+eps
            if eigenValues.min() < lowerBound:
                print("{0} out of bound\n".format(eigenValues.min()))
            if eigenValues.max() > upperBound:
                print("{0} out of bound\n".format(eigenValues.max()))
            x = int(file.split(".")[0])
            filePath = objectDir + "/" + workDir + ".csv"
            f = open(filePath, "a")
            s = ""
            for v in eigenValues:
                s = s + str(v) + ","
            f.write(s)
            f.write(str(labelList[x-1]))
            f.write("\n")

def generate_data(sourceDir, objectDir, normalized):
    # read mesh and segmentation data
    # get labels
    create_direcotry(objectDir)
    path = sourceDir + "/seg"
    files = os.listdir(path)
    total = 0
    for file in files:
        filePath = path + "/" + file
        if not os.path.isdir(filePath):
            total = max(int(file.split(".")[0]), total)

    print("total = %d" % total)
    labelList = np.zeros(total)
    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file):
            lable = get_segnum(path + "/" + file)
            x = int(file.split(".")[0])
            labelList[x-1] = lable

    # get maximum number of vertices, that is laplacian's dimension
    dimension = get_max_dim(sourceDir)

    workDir = "test"
    append_data_label(sourceDir, objectDir, workDir, labelList, normalized, dimension)
    workDir = "train"
    append_data_label(sourceDir, objectDir, workDir, labelList, normalized, dimension)

if __name__ == '__main__':
    cosegDir = "coseg"
    dataDir = "dataset/eig"
    dirs = os.listdir(cosegDir)
    for dir in dirs:
        print(dir)
        sourceDir = cosegDir + "/" + dir
        objectDir = dataDir + "/" + dir
        if os.path.isdir(sourceDir):
            generate_data(sourceDir, objectDir, True)