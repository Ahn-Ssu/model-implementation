import numpy as np
import csv
import os
from numpy.lib.function_base import average
import pandas as pd
from rdkit import Chem

# path = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/0_DehydroH_new_dataSet_fromCCEL.xlsx"
path  = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/1_[processed]QM9_deH_SMILES.xlsx"

id_target = np.array(pd.read_excel(path))

smiles = id_target #[:, 0]


def make_dilation_adj(
    originalAdj:np.ndarray,
    dilation:int = 1 ,
    selfLoop:bool=False
    )->np.ndarray:

    if dilation == 1:
        return originalAdj

    d_adj = originalAdj.copy()
    d_adj.fill(0)

    for idx, Vneibor in enumerate(originalAdj):
        for N, connectivity in enumerate(Vneibor):
            if connectivity:
                d_adj[idx] += originalAdj[N]


    # remove self loop
    if not selfLoop:
        for iter in range(len(d_adj)):
            d_adj[iter][iter] = 0


    return make_dilation_adj(d_adj, dilation=dilation-1)


def mat_power(
    origin_A:np.ndarray,
    power_A:np.ndarray,
    n:int=1
    )->np.ndarray:
    # print(n)

    power_A = power_A+ np.eye(len(power_A))
    if n == 1 :
        return power_A
    return mat_power(origin_A, np.dot(origin_A>0, power_A>0), n-1)   

def calc_coverage(
    src_arr:np.ndarray,
    n:int=1,
    loop=True):


    slot = np.zeros((len(src_arr),len(src_arr)), dtype=np.int32)
    for iter in range(n):
        slot = slot + mat_power(src_arr, src_arr, iter+1)
        
    

    if loop :
        return ((slot + np.eye(len(src_arr)))>0).mean()
    return (slot>0).mean()
    
conv = 4

path = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/"
if not os.path.exists(path+f'conv{conv} full covered data.csv'):
      with open(path+f'[receptive] conv{conv} full covered data.csv', mode='w') as f:
        myWriter = csv.writer(f)
while conv:
    c = 0 
    sum = []
    for aSMILES, deH in smiles:
        # aSMILES = 'CCCCCCCCCC'
        mol = Chem.MolFromSmiles(aSMILES)
        adj = Chem.GetAdjacencyMatrix(mol)

        # print()
        # print(aSMILES)
        # print(f'conv1 converage {calc_coverage(adj)}')
        sum.append(calc_coverage(adj,conv))

        if calc_coverage(adj,conv) == 1:
            c +=1
            with open(path+f'[receptive] conv{conv} full covered data.csv', mode='a') as f:  
                myWriter = csv.writer(f)
                myWriter.writerow([aSMILES, deH])
            # print(aSMILES, calc_coverage(adj, conv-1))
    #     print(f'conv2 converage {calc_coverage(adj,2)}')
    #     print(f'conv3 converage {calc_coverage(adj,3)}')
    #     print(f'conv4 converage {calc_coverage(adj,4)}')
    #     print(f'conv5 converage {calc_coverage(adj,5)}')
    #     print(f'conv6 converage {calc_coverage(adj,6)}')
    #     print(f'conv7 converage {calc_coverage(adj,7)}')
    #     print(f'conv8 converage {calc_coverage(adj,8)}')

    print(f'# of full coverage mol {c}')
    print(f'Receptive Coverage : Conv{conv} = {average(sum)}')

    if conv == 4:
        break
    if average(sum) == 1 :
        conv = 0
    else: 
        conv += 1 


# CCEL은 14층에서 1.0 완성  6층에서 95.3, 7층에서 98
# Receptive Coverage : Conv1 = 0.25438386702993493
# Receptive Coverage : Conv2 = 0.4878151959044368
# Receptive Coverage : Conv3 = 0.6850971630090275
# Receptive Coverage : Conv4 = 0.8204262219502966
# Receptive Coverage : Conv5 = 0.9059913768514544
# Receptive Coverage : Conv6 = 0.9531317665186555
# Receptive Coverage : Conv7 = 0.976653590381227
# Receptive Coverage : Conv8 = 0.9872514145181495
# Receptive Coverage : Conv9 = 0.9931812035250754
# Receptive Coverage : Conv10 = 0.9972223481620011
# Receptive Coverage : Conv11 = 0.9993042419790814
# Receptive Coverage : Conv12 = 0.9999213089445425
# Receptive Coverage : Conv13 = 0.9999890705114068
# Receptive Coverage : Conv14 = 1.0

# QM9은 6층에서 1.0 완성 4층에서 98, 5층에서 99.9 
# Receptive Coverage : Conv1 = 0.3627888185352817
# Receptive Coverage : Conv2 = 0.6797904461396128
# Receptive Coverage : Conv3 = 0.9079665022599999
# Receptive Coverage : Conv4 = 0.9865342181276132
# Receptive Coverage : Conv5 = 0.999253651335593
# Receptive Coverage : Conv6 = 1.0