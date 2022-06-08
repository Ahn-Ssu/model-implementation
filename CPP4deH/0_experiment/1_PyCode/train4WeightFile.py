import sys
sys.path.append("/home/ahn_ssu/CP2GN2/0_experiment/1_PyCode/")
import os
import numpy as np
import utils.ml
import utils.chem as chem
from tqdm import tqdm
from torch_geometric.data import DataLoader
# from sklearn.model_selection import train_test_split
from model.models import FNN, GCN, GIN, KAIST_GAT, SAGIN
import argparse
import csv 


import argparse
import socket 

import torch
import utils.ml
import utils.chem as chem
# from model.models import SAGE, GCN

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))


# Setting
paser = argparse.ArgumentParser()
args = paser.parse_args("")

# 1. Model 
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
args.batch_size = 32
args.init_lr = 0.0001 #default 0.001 
args.l2_coeff = 0.0
args.n_epochs = 100000

exp_label = f"[new64] KAIST_GAT (3+mean+2) | batch={args.batch_size} | {s.getsockname()[0][-3:]}.{socket.gethostname()} exp"

# Load dataset
print('Load molecular structures...')
dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/4_CCEL_data_(wo_B_NMe).xlsx"
dataSet = chem.load_dataset(dataPath, featrue_num= args.ex_dim)
print("Done!")

dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/KDD__test1_data_64.xlsx"
test_Set = chem.load_dataset(dataPath, featrue_num= args.ex_dim)


#GPU Using
DEVICE = utils.ml.GPU_check()
args.MODEL_NAME = 'SAGIN'
args.DEVICE = DEVICE
print("MODEL_NAME = {}, DEVICE = {}".format(args.MODEL_NAME,args.DEVICE))

model = args.model(args).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.l2_coeff)
criterion = torch.nn.L1Loss()

# train_data, validate_data, test_data = utils.ml.m_feat_normalize(train= dataSet[:, 1], valid=test_Set[:, 1], test1=test_Set[:, 1])#, test2=test2_data)
# # for Learning All dataSet
train_smiles = dataSet[:, 0]
train_loader = DataLoader(dataSet[:, 1], batch_size=args.batch_size, shuffle=True, drop_last=True)
train_targets = np.array([x.y.item() for x in dataSet[:,1]]).reshape(-1, 1) 

test_smiles = test_Set[:, 0]
test_loader = DataLoader(test_Set[:, 1], batch_size=64, shuffle=False)
test_targets = np.array([x.y.item() for x in test_Set[:,1]]).reshape(-1, 1) 


# Generate training and test datasets
# train : validation : test = 8 : 1 : 1
# train_data, unseen_data = train_test_split(dataSet,test_size=0.2, random_state=400)
# validate_data, test_data = train_test_split(unseen_data, test_size=0.5, random_state=400)

# train_smiles = train_data[:, 0]
# validate_smiles = validate_data[:, 0]
# test_smiles = test_data[:, 0]

# train_loader = DataLoader(train_data[:, 1], batch_size=args.batch_size, shuffle=True, drop_last=True)
# validate_loader = DataLoader(validate_data[:, 1], batch_size=100, drop_last=False)
# test_loader = DataLoader(test_data[:, 1], batch_size=100, drop_last=False)

# train_targets = np.array([x.y.item() for x in train_data[:,1]]).reshape(-1, 1)
# validate_targets = np.array([x.y.item() for x in validate_data[:,1]]).reshape(-1, 1)
# test_targets = np.array([x.y.item() for x in test_data[:,1]]).reshape(-1, 1)

from datetime import date, datetime, timezone, timedelta
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Train graph neural network (GNN)
print(exp_label, 'Train the GNN-based predictor...')
savePath = "/home/ahn_ssu/CP2GN2/4_weigthFile/"
exp_day = str(date.today())
savePath += exp_day+"/"
try:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
except OSError:
        print("Error: Cannot create the directory {}".format(savePath))

if not os.path.exists(savePath):
    with open(savePath, mode='w') as f:
        myWriter = csv.writer(f)
        myWriter.writerow(["epoch","tMAE","tR2"])
# print(test_targ;ets)
# input()
for i in tqdm(range(0, args.n_epochs)):
    utils.ml.train(model, optimizer, train_loader, criterion, args.DEVICE)

    if ((i % 2500)) == 0 :
        preds_t1 = utils.ml.test(model, test_loader, args.DEVICE)
        # print(preds_t1)
        # input()
        t1_mae = mean_absolute_error(test_targets, preds_t1)
        t1_r2 = r2_score(test_targets, preds_t1)
        
        saveName = savePath + exp_label+"(epoch"+str(i)+ ")_train.pth"

        with open(savePath+exp_label+"(monitor).csv", mode='a') as f:  
            myWriter = csv.writer(f)
            # myWriter.writerow([stage+1, idx+1, test_mae, test_mse, r2])
            myWriter.writerow([i+1, t1_mae, t1_r2])#, t2_mae, t2_r2])
        torch.save(model.state_dict(), saveName)