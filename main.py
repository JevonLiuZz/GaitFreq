"""
Author: Liu
Time: 2023.06.14
"""
import sys
import time
import yaml
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from FileRead import GaitDataLoader, DataSplit
from torch.utils.tensorboard import SummaryWriter
from model import GaitModel


def yaml_func(yaml_path):
    cfg = open(yaml_path, 'r', encoding='utf-8').read()
    conf = yaml.load(cfg, Loader=yaml.FullLoader)
    return conf


# class DimensionError(ValueError):
#     pass


def main(kwargs):
    train_cfg = yaml_func(kwargs.train_config_path)
    val_cfg = yaml_func(kwargs.validation_config_path)
    data_cfg = yaml_func(kwargs.data_config_path)

    torch.manual_seed(train_cfg["manual seed"])
    torch.cuda.manual_seed(train_cfg["manual seed"])

    kwargs.use_cuda = kwargs.use_cuda and torch.cuda.is_available()
    device = "cuda:0" if kwargs.use_cuda else "cpu"
    print(("Sorry! GPU is not contained." if device != "cuda:0" else "从未长大，但从未停止成长，从你的想象，到世界现象。\nGPU activated!"))

    writer = SummaryWriter(log_dir='./log/GaitLog')

    # --------------------- Data Processing --------------------
    dataFull, labelFull = DataSplit.get_data(train_cfg["data_path"], train_cfg["label_path"],
                                             data_cfg['dynamic window size'])
    trainCeil = int(np.floor(data_cfg['train percent']*dataFull.shape[0]))
    trainData = dataFull[:trainCeil, :, ...]
    trainLabel = labelFull[:trainCeil, ...]
    valData = dataFull[trainCeil:, :, ...]
    valLabel = labelFull[trainCeil:, ...]

    # --------------------- Load data & model --------------------
    trainDataSet = GaitDataLoader.GaitDataSet(trainData, trainLabel)
    # # Data Attribute
    # oneAllVarIndex, oneAllVarLabel = dataSet.__getitem__(27)
    # gaitTrainDataLen = trainDataSet.__len__()
    trainDataLoader = DataLoader(trainDataSet, batch_size=train_cfg["batch size"], shuffle=True, num_workers=0)
    valDataSet = GaitDataLoader.GaitDataSet(valData, valLabel)
    gaitValDataLen = valDataSet.__len__()
    valDataLoader = DataLoader(valDataSet, batch_size=val_cfg["batch size"], shuffle=True, num_workers=0)

    processModel = GaitModel.GaitModelNN().to(device)

    # --------------------- Train setting --------------------
    Optimizer = getattr(sys.modules['torch.optim'], train_cfg['Optimizer'])
    opti = Optimizer(processModel.parameters(), lr=train_cfg['learning rate'], weight_decay=train_cfg['weight decay'])
    sch = train_cfg.get('scheduler', None)
    if sch is None:
        scheduler = None
        print(f"Scheduler failed!")
    else:
        Scheduler = getattr(sys.modules['torch.optim.lr_scheduler'], sch.pop('name'))
        scheduler = Scheduler(opti, **sch)
    gaitLoss = torch.nn.MSELoss(reduction='mean').to(device)

    # --------------------- Train  & Validation --------------------
    # Train Flow (& Validation embedded)
    for epoch in range(train_cfg["epoch size"]):
        epoch_start = time.time()
        processModel.train()
        lossTotal = 0
        batchCnt = 0
        for data, label in trainDataLoader:
            batchCnt += 1
            # dataReshapeDim = data[0, :, ...].shape
            # data = data.reshape(dataReshapeDim)
            # labelReshapeDim = label[0, :, ...].shape
            # label = label.reshape(labelReshapeDim)
            data = data.to(device)
            label = label.to(device)

            trainResult = processModel(data)
            lossBatch = gaitLoss(trainResult, label)
            lossBatch.requires_grad_(True)
            lossBatch.backward()

            lossTotal += lossBatch
            processModel.zero_grad()
            opti.step()
            # writer.add_scalar(f'Epoch_{epoch+1}: Batch Loss Curve', lossBatch, global_step=batch)
        epoch_end = time.time()
        epochTimeInterval = epoch_end - epoch_start
        #  saveGaitModel()  # need to be realized
        torch.save(processModel, f'outPuts/gaitPthDir/myProcessGaitModel_Epoch_{epoch+1}.pth')
        writer.add_scalar('Epoch Loss Curve', lossTotal, global_step=epoch)
        if (epoch % train_cfg['epoch_info_interval']) == 0:
            print(f"| epoch: {epoch+1} | process time: {epochTimeInterval} | train_total_loss {lossTotal / batchCnt}")

        if epoch % int(train_cfg['scheduler para']) == 4:  # verify whether to add
            scheduler.step()

        # --------------------- Validation --------------------
        processModel.eval()
        # # validation process, model compare & save need to be added
        accuracyTotal = 0

        with torch.no_grad():
            for data, label in valDataLoader:
                # dataReshapeDim = data[0, :, ...].shape
                # data = data.reshape(dataReshapeDim)
                # labelReshapeDim = label[0, :, ...].shape
                # label = label.reshape(labelReshapeDim)
                dataWindow = data.to(device)
                labelWindow = label.to(device)

                validationResult = processModel(dataWindow)
                validationTarget = validationResult.argmax(1)
                labelWindowTarget = labelWindow.argmax(1)
                # zxc = (validationTarget == labelWindowTarget).sum()
                accuracyTotal += ((validationTarget == labelWindowTarget).sum())
                # if validationTarget == labelWindowTarget:
                #     accuracyTotal += 1

        accuracyRatio = accuracyTotal / gaitValDataLen
        print(f"| epoch: {epoch+1} | accuracy: {accuracyRatio}")

    # Log by SummaryWriter
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Fleeter Gait Freq Pred', description='PyTorch realization')
    parser.add_argument('--train_config_path', type=str, default='./CFG/train.yaml',
                        help='Train config args!')
    parser.add_argument('--validation_config_path', type=str, default='./CFG/validation.yaml',
                        help='Validation config args!')
    parser.add_argument('--data_config_path', type=str, default='./CFG/dataCFG.yaml',
                        help='Data load config args!')
    parser.add_argument('--result_directory', type=str, default="./outPuts/results")
    parser.add_argument('--use_cuda', default=True, help="using CUDA for training")
    args = parser.parse_args()
    main(args)
