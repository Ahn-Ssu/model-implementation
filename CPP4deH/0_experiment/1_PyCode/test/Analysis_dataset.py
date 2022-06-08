
import numpy as np
import pandas as pd
from tqdm import tqdm
# from rdkit.Chem.rdMolDescriptors import CalcNumAliphaticRings, CalcnumAromaticRings,CalcNumSaturatedRing,CalcNumRings 
import rdkit 
from rdkit import Chem
from rdkit.Chem import Descriptors as desc


# dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/DehydroH_new_dataSet_fromCCEL.xlsx"
dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/1_[processed]QM9_deH_SMILES.xlsx"
dataset = np.array(pd.read_excel(dataPath))

result = []
# dataset[ ,0] = SMILES
# dataset[ ,1] = Label
for idx in tqdm(range(dataset.shape[0])):

    mol = Chem.MolFromSmiles(dataset[idx, 0])
    
    # alphaticRing = rdkit.Chem.Lipinski.NumAliphaticRings(mol)
    # aromaticRing = rdkit.Chem.Lipinski.NumAromaticRings(mol)
    # saturatedRing = rdkit.Chem.Lipinski.NumSaturatedRings(mol)
    # ringCount = RingCount(mol)

    MolWt = rdkit.Chem.Descriptors.MolWt(mol)
    HeavyAtomMolWt = rdkit.Chem.Descriptors.HeavyAtomMolWt(mol)
    NumValenceElectrons = rdkit.Chem.Descriptors.NumValenceElectrons(mol)
    FractionCSP3 = rdkit.Chem.Lipinski.FractionCSP3(mol)
    HeavyAtomCount = rdkit.Chem.Lipinski.HeavyAtomCount(mol)
    NHOHCount = rdkit.Chem.Lipinski.NHOHCount(mol)
    NOCount = rdkit.Chem.Lipinski.NOCount(mol)
    NumAliphaticCarbocycles = rdkit.Chem.Lipinski.NumAliphaticCarbocycles(mol)
    NumAliphaticHeterocycles = rdkit.Chem.Lipinski.NumAliphaticHeterocycles(mol)
    NumAliphaticRings = rdkit.Chem.Lipinski.NumAliphaticRings(mol)
    NumAromaticCarbocycles = rdkit.Chem.Lipinski.NumAromaticCarbocycles(mol)
    NumAromaticHeterocycles = rdkit.Chem.Lipinski.NumAromaticHeterocycles(mol)
    NumAromaticRings = rdkit.Chem.Lipinski.NumAromaticRings(mol)
    NumHAcceptors = rdkit.Chem.Lipinski.NumHAcceptors(mol)
    NumHDonors = rdkit.Chem.Lipinski.NumHDonors(mol)
    NumHeteroatoms = rdkit.Chem.Lipinski.NumHeteroatoms(mol)
    NumRotatableBonds = rdkit.Chem.Lipinski.NumRotatableBonds(mol)
    RingCount = rdkit.Chem.Lipinski.RingCount(mol)
    MolMR = rdkit.Chem.Crippen.MolMR(mol)
    CalcNumBridgeheadAtom = rdkit.Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    ## 2. non - Highlight Feature from CCEL 
    ## 12 EA
    ExactMolWt = rdkit.Chem.Descriptors.ExactMolWt(mol)
    NumRadicalElectrons = rdkit.Chem.Descriptors.NumRadicalElectrons(mol)
    MaxPartialCharge = rdkit.Chem.Descriptors.MaxPartialCharge(mol)
    MinPartialCharge = rdkit.Chem.Descriptors.MinPartialCharge(mol)
    MaxAbsPartialCharge = rdkit.Chem.Descriptors.MaxAbsPartialCharge(mol)
    MinAbsPartialCharge = rdkit.Chem.Descriptors.MinAbsPartialCharge(mol)
    NumSaturatedCarbocycles = rdkit.Chem.Lipinski.NumSaturatedCarbocycles(mol)
    NumSaturatedHeterocycles = rdkit.Chem.Lipinski.NumSaturatedHeterocycles(mol)
    NumSaturatedRings = rdkit.Chem.Lipinski.NumSaturatedRings(mol)
    MolLogP = rdkit.Chem.Crippen.MolLogP(mol)
    CalcNumAmideBonds = rdkit.Chem.rdMolDescriptors.CalcNumAmideBonds(mol)
    CalcNumSpiroAtoms = rdkit.Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)

    recom = [MolWt, HeavyAtomMolWt, NumValenceElectrons, FractionCSP3, HeavyAtomCount, NHOHCount, NOCount, NumAliphaticCarbocycles, NumAliphaticHeterocycles, NumAliphaticRings, NumAromaticCarbocycles, NumAromaticHeterocycles, NumAromaticRings, NumHAcceptors, NumHDonors, NumHeteroatoms, NumRotatableBonds, RingCount, MolMR, CalcNumBridgeheadAtom]
    notRecom = [ExactMolWt, NumRadicalElectrons, MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge, MinAbsPartialCharge, NumSaturatedCarbocycles, NumSaturatedHeterocycles, NumSaturatedRings, MolLogP, CalcNumAmideBonds, CalcNumSpiroAtoms]
    recom += notRecom

    result.append([dataset[idx,0]] + recom +[dataset[idx, 1] ])


info_df = pd.DataFrame(result, columns=["SMILES",
"MolWt","HeavyAtomMolWt","NumValenceElectrons","FractionCSP3","HeavyAtomCount","NHOHCount","NOCount","NumAliphaticCarbocycles","NumAliphaticHeterocycles","NumAliphaticRings","NumAromaticCarbocycles","NumAromaticHeterocycles","NumAromaticRings","NumHAcceptors","NumHDonors","NumHeteroatoms","NumRotatableBonds","RingCount","MolMR","CalcNumBridgeheadAtom",
"ExactMolWt","NumRadicalElectrons","MaxPartialCharge","MinPartialCharge","MaxAbsPartialCharge","MinAbsPartialCharge","NumSaturatedCarbocycles","NumSaturatedHeterocycles","NumSaturatedRings","MolLogP","CalcNumAmideBonds","CalcNumSpiroAtoms",
"deH"])

savePath = "/home/ahn_ssu/CP2GN2/2_data/4_info/"
info_df.to_excel(savePath+"QM9-Info.xlsx")


# df1 = pd.read_excel("/content/1stData-Info.xlsx")

# plt.figure(figsize=(60,40))
# plt.suptitle("1st Data Set", fontsize=20)
# for idx, label in enumerate(feature):
#   plt.subplot(8,4,idx+1)
#   plt.title(label)
#   plt.scatter(df1[label], df1['deH'],c=df1['deH'], cmap=plt.cm.cool)
#   plt.grid()
# plt.savefig('filename.png', dpi=300)