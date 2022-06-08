import argparse
import socket 
import torch

import utils.ml
import utils.chem as chem

from model.models import FNN, GCN, GIN, SAGIN ,KAIST_GAT

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))

# Setting
paser = argparse.ArgumentParser()
args = paser.parse_args("")


# 1. Model 
args.model = SAGIN
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
args.batch_size = 64
args.init_lr = 0.0001 #default 0.001 
args.l2_coeff = 0.0
args.n_epochs = 30000
args.optimizer = torch.optim.Adam
args.criterion = torch.nn.L1Loss

# 0. Exp case (for CV record)
exp_label = f"[w'o NMe] GCN(4) | mean pooling + linear(2) | optim={args.optimizer.__name__}, criterion={args.criterion.__name__}, aggr={args.convAggr}, lr={args.init_lr}, Channel={args.hidden_dim}, batch={args.batch_size} | {s.getsockname()[0][-3:]}.{socket.gethostname()} exp"

# Load datase
print('Load molecular structures...')
dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/2_EPI_PHYSPROP dataSet/KDD BenchSet/BP_train.xlsx"
dataSet = chem.load_dataset(dataPath, featrue_num= args.ex_dim)

dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/2_EPI_PHYSPROP dataSet/KDD BenchSet/BP_test.xlsx"
test1_Set = chem.load_dataset(dataPath, featrue_num= args.ex_dim)

# dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/KDD__test2_QM9_data_5975.xlsx"
# test2_Set = chem.load_dataset(dataPath, featrue_num= args.ex_dim)


#GPU Using
DEVICE = utils.ml.GPU_check()
args.MODEL_NAME = 'SAGE' 
args.DEVICE = DEVICE
print("MODEL_NAME = {}, DEVICE = {}".format(args.MODEL_NAME,args.DEVICE))

# CrossValidation call
utils.ml.crossValidation(data = dataSet[:,1],test1_data=test1_Set[:,1], n_splits=5, args=args, savePath = "/home/ahn_ssu/CP2GN2/3_output", case=exp_label)
