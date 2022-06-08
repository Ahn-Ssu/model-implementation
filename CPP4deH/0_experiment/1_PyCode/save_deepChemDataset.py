import deepchem
# from deepchem.trans.transformers import 
import numpy as np
import torch
import pandas as pd
import deepchem as dc 
from deepchem.molnet import *
from tqdm import tqdm



# http://moleculenet.ai/datasets-1 데이터셋 있는 위치 
# Regression 
# QM7/QM7b, QM8, QM9, ESOL, FreeSolv, Lipophilcity, PDBbind, BACE 
## load_qm7()
## load_qm8()
## load_qm9()
## load_delaney() : ESOL 1128 https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#contributing-a-new-dataset-to-moleculenet
## load_sampl() : FreeSolv https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html#contributing-a-new-dataset-to-moleculenet 
## load_lipo() 
## load_pdbbind() 
## load_bace_regression() 

# Classification
# PCBA, MUV, HIV, BACE, BBBP, Tox21, ToxCast, SIDER, ClinTox

# '3D Coordinate 
# QM7/QM7b, QM8, QM9:Strcutre / PDB



# QM7 load_qm7() 
def load_dataSet2local(dataType = "reg"):

    
    # regression problem 
    if dataType == "reg":
        label_List = ["QM7", "QM8", "QM9", "ESOL", "FreeSolv", "Lipophilcity", "PDBbind", "BACE"]
        function_list = [load_qm7,load_qm8,load_qm9,load_delaney,load_sampl,load_lipo ,load_pdbbind ,load_bace_regression]
    elif dataType == "classifi":
        # classification problem
        label_List = ["BACE", "BBBP", "ClinTox", "HIV", "MUV", "PCBA", "Tox21", "ToxCast", "SIDER"]
        function_list = [load_bace_classification, load_bbbp, load_clintox, load_hiv, load_muv, load_pcba, load_tox21, load_toxcast, load_sider]
        # [14:33:54] WARNING: not removing hydrogen atom without neighbors - Tox21 경고 


    for oneSet_label, oneSet_Call in tqdm(zip(label_List, function_list)):
        print(oneSet_label)
        taskName, datasets, transformers = oneSet_Call(featurizer='GraphConv')
        df = None
        for idx, dataset in enumerate(datasets):
            
            if df is None:
                df = dataset.to_dataframe()
            else:
                df = pd.concat([df, dataset.to_dataframe()], axis=0)
            
        path = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/" + dataType +"_"
        path += oneSet_label + ".xlsx"
        df.to_excel(path)



# Failed to featurize datapoint 192, ('/tmp/v2013-core/4gqq/4gqq_ligand.sdf', '/tmp/v2013-core/4gqq/4gqq_pocket.pdb'). Appending empty array
# Exception message: 'tuple' object has no attribute 'GetAtoms'


# QM7_MAT_UTL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.mat"
# QM7_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv"
# QM7B_MAT_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7b.mat"
# GDB7_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb7.tar.gz"
# QM7_TASKS = ["u0_atom"]

# GDB8_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz"
# QM8_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv"
# QM8_TASKS = [
#     "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0",
#     "f2-PBE0", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM",
#     "f1-CAM", "f2-CAM"
# ]

# GDB9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
# QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
# QM9_TASKS = [
#     "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
#     "h298", "g298"
# ]

# DELANEY_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
# DELANEY_TASKS = ['measured log solubility in mols per litre']

# SAMPL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
# SAMPL_TASKS = ['expt']

# LIPO_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
# LIPO_TASKS = ['exp']

# 얘는 좀 구성이 다름
# DATASETS_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
# PDBBIND_URL = DATASETS_URL + "pdbbindv2019/"
# PDBBIND_TASKS = ['-logKd/Ki']

# BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
# BACE_REGRESSION_TASKS = ["pIC50"]
# BACE_CLASSIFICATION_TASKS = ["Class"]

if __name__ == '__main__':
    load_dataSet2local("reg")
    print("\t\t**Clear**")

