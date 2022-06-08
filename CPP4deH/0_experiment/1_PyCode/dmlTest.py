import argparse
import socket

from tqdm import tqdm 

import utils.ml
import utils.chem as chem
from model.models import SAGE, GCN

import numpy
import random
import pandas
import torch
import xgboost as xgb
import utils.mol_dml
# import utils.chem as chem
from torch_geometric.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))

# Setting
paser = argparse.ArgumentParser()
args = paser.parse_args("")


# 1. Model 
args.model = SAGE
args.input_dim = utils.chem.n_atom_feats
args.hidden_dim = 128
args.dim_emb = 1024
args.out_dim = 1
args.head_num = 4
args.ex_dim = 20
args.convNorm = False
args.skipConcat = True
args.skipCompress = False
args.convAggr = 'add'

# 2. learing 
args.batch_size = 17
args.init_lr = 0.0001 #default 0.001 
args.l2_coeff = 0.0
args.n_epochs = 30000


# Load dataset
print('Load molecular structures...')
dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/3_CCEL_data_(wo_B).xlsx"
dataSet = chem.load_dataset(dataPath, featrue_num= args.ex_dim)
random.shuffle(dataSet)
smiles = [x[0] for x in dataSet]
mols = [x[1] for x in dataSet]


# Generate training and test datasets
n_train = int(0.8 * len(dataSet))
train_data = mols[:n_train]
test_data = mols[n_train:]
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
emb_loader = DataLoader(train_data, batch_size=args.batch_size)
test_loader = DataLoader(test_data, batch_size=args.batch_size)
train_smiles = numpy.array(smiles[:n_train]).reshape(-1, 1)
test_smiles = numpy.array(smiles[n_train:]).reshape(-1, 1)
train_targets = numpy.array([x.y.item() for x in train_data]).reshape(-1, 1)
test_targets = numpy.array([x.y.item() for x in test_data]).reshape(-1, 1)


# Model configuration
args.DEVICE = utils.ml.GPU_check()
# args.DEVICE = 'cpu'
emb_net = SAGE(args).to(args.DEVICE)
optimizer = torch.optim.Adam(emb_net.parameters(), lr=args.init_lr, weight_decay=args.l2_coeff)


# Train GNN-based embedding network
print('Train the GNN-based embedding network...')
stage_indicator = "Epoch"
for i in tqdm(range(0, args.n_epochs), desc=stage_indicator):
    train_loss = utils.mol_dml.train(emb_net, optimizer, train_loader, args.DEVICE)
    print('Epoch [{}/{}]\tTrain loss: {:.4f}'.format(i + 1, args.n_epochs, train_loss), end='\r')


# Generate embeddings of the molecules
train_embs = utils.mol_dml.test(emb_net, emb_loader, args.DEVICE)
test_embs = utils.mol_dml.test(emb_net, test_loader, args.DEVICE)
train_emb_results = numpy.concatenate([train_embs, train_smiles, train_targets], axis=1).tolist()
test_emb_results = numpy.concatenate([test_embs, test_smiles, test_targets], axis=1).tolist()
df = pandas.DataFrame(train_emb_results)
df.to_excel('/home/ahn_ssu/CP2GN2/0_experiment/1_PyCode/res/embs_train.xlsx', header=None, index=None)
df = pandas.DataFrame(test_emb_results)
df.to_excel('/home/ahn_ssu/CP2GN2/0_experiment/1_PyCode/res/embs_test.xlsx', header=None, index=None)


# Train XGBoost using the molecular embeddings
print('Train the XGBoost regressor...')
model = xgb.XGBRegressor(max_depth=8, n_estimators=300, subsample=0.8)
model.fit(train_embs, train_targets, eval_metric='mae')
preds = model.predict(test_embs).reshape(-1, 1)
test_mae = mean_absolute_error(test_targets, preds)
r2 = r2_score(test_targets, preds)
print('Test MAE: {:.4f}\tTest R2 score: {:.4f}'.format(test_mae, r2))


# Save prediction results
pred_results = list()
for i in range(0, preds.shape[0]):
    pred_results.append([test_smiles[i], test_targets[i].item(), preds[i].item()])
df = pandas.DataFrame(pred_results)
df.columns = ['smiles', 'true_y', 'pred_y']
df.to_excel('res/preds/preds__dml.xlsx', index=False)
