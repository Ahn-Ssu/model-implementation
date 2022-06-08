from multiprocessing.synchronize import SemLock
import numpy as np
import pandas as pd
import torch
import copy
import os
import argparse
import utils.ml
import utils.chem as chem
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from model.models import SAGIN,GCN,GIN,KAIST_GAT


# Setting
paser = argparse.ArgumentParser()
args = paser.parse_args("")

# 0. Exp case (for CV record)
exp_label = "new pred - KAIST_GAT(3)"

args.model = KAIST_GAT
args.input_dim = utils.chem.n_atom_feats
args.hidden_dim = 128
args.out_dim = 1
args.head_num = 4
args.ex_dim = 20
args.convNorm = False
args.skipConcat = True
args.skipCompress = False
args.convAggr = 'add'

# 2. learing 
args.batch_size = 128
args.init_lr = 0.0001 #default 0.001 
args.l2_coeff = 0.0
args.n_epochs = 100000




model_PATH = "/home/ahn_ssu/CP2GN2/4_weigthFile/2022-04-28/[CCEL] KAIST_GAT (3+mean+2) | batch=32 | 162.nipa2019-0458 exp(epoch40000)_train.pth"
dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/[predTarget]현우님 요청, new materials.xlsx"
             
#GPU Using
DEVICE = 'cpu' #utils.ml.GPU_check()
args.MODEL_NAME = 'SAGIN'
args.DEVICE = DEVICE
print("MODEL_NAME = {}, DEVICE = {}".format(args.MODEL_NAME,args.DEVICE))

model = args.model(args).to(DEVICE)
model.load_state_dict(copy.deepcopy(torch.load(model_PATH)))
# model.share_memory()

optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.l2_coeff)
criterion = torch.nn.L1Loss()

from datetime import date, datetime, timezone, timedelta

savePath = "/home/ahn_ssu/CP2GN2/5_predict/"
exp_day = str(date.today())
savePath += exp_day+"/"
try:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
except OSError:
        print("Error: Cannot create the directory {}".format(savePath))

# 학습한 데이터로 norm을 해주어야 예측을 제대로 하지요?
# trainPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/4_CCEL_data_(wo_B_NMe).xlsx"
# trainSet = chem.load_dataset(trainPath, featrue_num= args.ex_dim)


data = pd.read_excel(dataPath,index_col=False, header=None)
# data = pd.DataFrame(data.to_numpy().reshape(-1,23))
print(f'data shape:{data.shape}')
# print(len(data))
# exit()

count = 0
# # for prediting All dataSet 
def run(idx:int):
    global count
    # Load dataset
    print(f'Load molecular structures... {idx}', end="\r")
    target = data[idx].dropna()
    # print(target.shape, count)
    # count += len(target)
    dataSet = chem.load_dataset4predict(target)

    test_smiles = dataSet[:, 0]

    # train, _, test_data  = utils.ml.m_feat_normalize(train=trainSet[:, 1], valid=trainSet[:, 1], test1=dataSet[:, 1])#, test2=test2_data)
    test_loader = DataLoader(dataSet[:, 1], batch_size=256, drop_last=False)
    # test_loader = DataLoader(dataSet[:, 1], batch_size=256, drop_last=False)

    # Test the trained GNN
    preds = utils.ml.test(model, test_loader , DEVICE)

    # Save prediction results
    pred_results = list()
    for i in range(0, preds.shape[0]):
        pred_results.append([test_smiles[i], preds[i].item()])
    df = pd.DataFrame(pred_results)
    df.columns = ['smiles', 'pred_y']
    df.to_excel(savePath + exp_label + f"{idx}.xlsx", index=False)

from multiprocessing import Pool


run(0)
# # # run(1)
# # # run(2)
# # # run(3)
# # # run(4)
# jobPool = Pool(processes=os.cpu_count())
# jobPool.map(run, range(0,48))
# jobPool.close()
# print("Main thread is done for run function[predict]")
# print(count)

files = os.listdir(savePath)



for idx, one in enumerate(files):
    print(f'now idx:{idx}/{len(files)}, file name:{one}', end="\r")
    if not idx:
        data = pd.read_excel(savePath+one,index_col=False, header=None)
    else:
        data = pd.concat([data, pd.read_excel(savePath+one,index_col=False, header=None)],axis=1)


print(data.shape)

data.to_excel(savePath + exp_label + "total.xlsx", index=False)