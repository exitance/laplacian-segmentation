import os
import numpy as np

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

def ascending_eigenValue(M):
    eigenvalue, _ = np.linalg.eig(M)
    return np.sort(eigenvalue)

def create_direcotry(path):
    folder = os.path.exists(path)

    if not folder:
        # recursively
        os.makedirs(path)
    else:
        print("folder %s exists" % (path))

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

def write_data_label(sourceDir, objectDir, workDir, labelList, normalized):
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
            eigenValues = ascending_eigenValue(laplacian)
            x = int(file.split(".")[0])
            filePath = objectDir + "/" + workDir + "/" + str(x) + ".in"
            f = open(filePath, "w")
            s = ""
            for v in eigenValues:
                s = s + str(v) + " "
            f.write(s)
            f.write("\n")
            f.write(str(labelList[x-1]))

def generate_eigenValue_data(sourceDir, objectDir, normalized):
    # read mesh and segmentation data
    # get labels
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

    workDir = "test"
    create_direcotry(objectDir + "/" + workDir)
    write_data_label(sourceDir, objectDir, workDir, labelList, normalized)
    workDir = "train"
    create_direcotry(objectDir + "/" + workDir)
    write_data_label(sourceDir, objectDir, workDir, labelList, normalized)
    files = os.listdir(path)


if __name__ == '__main__':
    
    cosegDir = "coseg"
    # dataDir = "data"
    dataDir = "data_normalized"
    dirs = os.listdir(cosegDir)
    for dir in dirs:
        print(dir)
        sourceDir = cosegDir + "/" + dir
        objectDir = dataDir + "/" + dir
        if os.path.isdir(sourceDir):
            generate_eigenValue_data(sourceDir, objectDir, True)