import numpy as np
import torch
import utils.ml
import utils.chem as chem
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from utils.models import GCN, eGCN, NeGCN, SNeGCN, GNeGCN, LEGNeGCN, MFGNeGCN, ClusterGNeGCN, SAGEGNeGCN, ChebGNeGCN, ATTGNeGCN, InceptlikeGCN, LW_inceptLikeGCN
import argparse


import os



rss_dir = "/home/ahn_ssu/CP2GN2/2_data/3_forBenchmark/1_deepChemNorm/"

files = os.listdir(rss_dir)


print("Checked files list. ->",files)
print("\n")

DEVICE = utils.ml.GPU_check()
for idx, oneData in enumerate(files):
        
    # Setting
    paser = argparse.ArgumentParser()
    args = paser.parse_args("")

    # 0. Exp case (for CV record)
    exp_label = "[Benchmark]"+oneData+", C=256 L1Loss, Summation, 20 mol.F | H=4, Summation GAT:KAIST"

    # 1. Model 
    # GCN, eGCN, NeGCN, SNeGCN, GNeGCN, 
    # LEGNeGCN, MFGNeGCN, ClusterGNeGCN, TAGGNeGCN, SAGEGNeGCN, ChebGNeGCN,
    #  ATTGNeGCN, InceptlikeGCN, LW_inceptLikeGCN
    args.model = ATTGNeGCN
    args.input_dim = utils.chem.n_atom_feats
    args.hidden_dim = 256
    args.out_dim = 1
    args.head_num = 4
    args.ex_dim = 20

    # 2. learing 
    args.batch_size = 17
    args.init_lr = 0.0001 #default 0.001 
    args.l2_coeff = 0.0
    args.n_epochs = 5000

    # Load dataset
    print('Load molecular structures...')
    dataPath = rss_dir+oneData
    dataSet = chem.load_dataset_for_benchmark(dataPath, args.ex_dim)

    #GPU Using
    args.MODEL_NAME = 'eGCN' 
    args.DEVICE = DEVICE
    print("MODEL_NAME = {}, DEVICE = {}".format(args.MODEL_NAME,args.DEVICE))

    utils.ml.crossValidation(data = dataSet[:,1], n_splits=5, args=args, savePath = "/home/ahn_ssu/CP2GN2/3_output", case=exp_label)