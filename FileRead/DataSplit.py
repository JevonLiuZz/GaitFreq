import torch
import numpy as np
from scipy.io import loadmat


def dynamic_window(data_seq, length, index_start):
    # procData = data_seq[:, :, index_start:index_start+length]
    procData = data_seq[:, index_start:index_start+length]
    return procData


def get_data(data_path, label_path, window_size):
    data = loadmat(data_path)
    label = loadmat(label_path)
    leftPos = data["POS"][:, 0]
    rightPos = data["POS"][:, 1]
    leftAcc = data["ACC"][:, -2]
    rightAcc = data["ACC"][:, -1]
    leftVel = data["VEL"][:, -2]
    rightVel = data["VEL"][:, -1]

    dimension_diff = max(np.shape(leftPos)[0], np.shape(leftVel)[0], np.shape(leftAcc)[0]) \
                     - min(np.shape(leftPos)[0], np.shape(leftVel)[0], np.shape(leftAcc)[0])
    leftPos = leftPos[dimension_diff:]
    rightPos = rightPos[dimension_diff:]

    PosCat = np.c_[leftPos, rightPos]
    VelCat = np.c_[leftVel, rightVel]
    AccCat = np.c_[leftAcc, rightAcc]

    dataRaw = np.c_[PosCat, VelCat, AccCat]
    dataRaw = dataRaw.T
    # dataRaw = dataRaw.reshape((1,) + dataRaw.shape)

    labelRaw = label["label"][:, :]
    labelRaw = labelRaw.T
    labelUniq = np.unique(labelRaw)
    oneHotLabel = torch.as_tensor(np.zeros((labelUniq.shape[0], labelRaw.shape[1])))
    for i in range(labelRaw.shape[1]):
        index = labelRaw[:, i].item()
        oneHotLabel[index, i] = 1
    # labelRaw = oneHotLabel.reshape((1,) + oneHotLabel.shape)
    labelRaw = oneHotLabel

    dataFrameRaw = []
    labelFrameRaw = []
    dataFrame = []
    labelFrame = []
    # for j in range(dataRaw[0, 0, :].shape[-1] - window_size):
    for j in range(dataRaw[0, :].shape[-1] - window_size):
        dataWindow = dynamic_window(dataRaw, window_size, j)
        # labelWindow = np.array(labelRaw[:, :, j + window_size])
        labelWindow = np.array(labelRaw[:, j + window_size])
        dataFrameRaw.append(dataWindow)
        labelFrameRaw.append(labelWindow)
    if np.size(dataFrameRaw, 0) == np.size(labelFrameRaw, 0):
        frameDimension = np.size(dataFrameRaw, 0)
    else:
        frameDimension = 0
        print(f"Data frame dimension is not equal to the dimension of Label frame!")

    indexGait = [i for i in np.arange(0, frameDimension, 1)]
    np.random.shuffle(indexGait)
    for i in indexGait:
        dataFrame.append(dataFrameRaw[i])
        labelFrame.append(labelFrameRaw[i])
    dataFrameProcessed = torch.as_tensor(np.array(dataFrame), dtype=torch.float)
    labelFrameProcessed = torch.as_tensor(np.array(labelFrame), dtype=torch.float)
    return dataFrameProcessed, labelFrameProcessed
