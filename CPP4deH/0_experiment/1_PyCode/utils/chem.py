import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from mendeleev import get_table
from sklearn import preprocessing
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors as desc
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from rdkit.Chem import MolSurf
# from rdkit.Chem.Descriptors import ExactMolWt


elem_feat_names = ['atomic_number','atomic_weight', 'atomic_radius', 'atomic_volume', 'dipole_polarizability',
                   'vdw_radius', 'en_pauling','boiling_point', 'electron_affinity', 
                   'en_allen', 'en_ghosh','mulliken_en','NumberofNeutrons','NumberofElectrons','NumberofProtons']
n_atom_feats = len(elem_feat_names)+5


# Load node Feature Table 
def get_elem_feats():
    # 기존 Preprocessing 을 수행하지 않은 데이터를 그대로 사용하던 코드 
    # tb_atom_feats = get_table('elements')
    # elem_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[elem_feat_names]))

    tb_atom_feats = pd.read_excel("/home/ahn_ssu/CP2GN2/2_data/2_preprocess/final_mendel_byCCEL.xlsx")
    elem_feats = np.array(tb_atom_feats[elem_feat_names])

    ## RDKit atom Feature extraction 


    return elem_feats

# Load smiles Data 
def load_dataset(path_user_dataset, featrue_num=20):
    elem_feats = get_elem_feats()
    list_mols = np.zeros((1,2))
    id_target = np.array(pd.read_excel(path_user_dataset, header=None))

    for i in tqdm(range(0, id_target.shape[0])):
        mol = smiles_to_mol_graph(elem_feats, id_target[i, 0], idx=i, target=id_target[i, 1], feature_num=featrue_num)
        # mol = smiles_to_ECFP(elem_feats, id_target[i, 0], idx=i, target=id_target[i, 1])
        
        
        if mol is not None:
          sample = np.array([id_target[i, 0], mol], dtype=object)
          list_mols = np.append(list_mols,
                                np.expand_dims(sample, axis=0),
                                axis=0)

    return list_mols[1:]


# Load smiles Data for predict
def load_dataset4predict(path_user_dataset, featrue_num=20):
    elem_feats = get_elem_feats()
    list_mols = np.zeros((1,2))

    if isinstance(path_user_dataset, str):
        id_target = np.array(pd.read_excel(path_user_dataset))
    elif isinstance(path_user_dataset, pd.DataFrame) or isinstance(path_user_dataset, pd.core.series.Series):
        id_target = path_user_dataset.to_numpy()

    for i in tqdm(range(0, id_target.shape[0])):
        mol = smiles_to_mol_graph(elem_feats, id_target[i], idx=i, target=0, feature_num=featrue_num)
        
        if mol is not None:
          sample = np.array([id_target[i], mol], dtype=object)
          list_mols = np.append(list_mols,
                                np.expand_dims(sample, axis=0),
                                axis=0)

    return list_mols[1:]

# for Benchmark 
def load_dataset_for_benchmark(dataset_name, featrue_num=20):
    elem_feats = get_elem_feats()
    # list_mols = list()
    list_mols = np.zeros((1,2))

    if "xlsx" in dataset_name:
        df = pd.read_excel(dataset_name)
        id_target = np.array(df[['ids','y']])
    else :
        df = pd.read_csv(dataset_name)
        id_target = np.array(df)
    

    for i in tqdm(range(0, id_target.shape[0])):
        mol = smiles_to_mol_graph(elem_feats, id_target[i, 0], idx=i, target=id_target[i, 1],feature_num=featrue_num)
        if mol is not None:
          sample = np.array([id_target[i, 0], mol], dtype=object)
          list_mols = np.append(list_mols,
                                np.expand_dims(sample, axis=0),
                                axis=0)
    
    return list_mols[1:]

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

# Convert smiles to torch embedding molecular features
def smiles_to_mol_graph(elem_feats, smiles, idx, target, feature_num = 20):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        # adding Identity Matrix 
        # adj_mat = adj_mat + numpy.eye(len(adj_mat)) 
        # adj_mat = make_dilation_adj(adj_mat, 3)
        atom_feats = list()
        bonds = list()
        Identity = list()
        

        # for atom in mol.GetAtoms():
        #     atom_feats.append(elem_feats[atom.GetAtomicNum() - 1, :])

        newfeats=list()
        for atom in mol.GetAtoms():
            atom_feats.append(elem_feats[atom.GetAtomicNum() - 1, :])
            newfeats.append([atom.GetTotalNumHs(),atom.IsInRing()*1,atom.GetTotalDegree(),
                             atom.GetTotalValence(),atom.GetIsAromatic()*1])

        # scaler = preprocessing.StandardScaler()
        # rd_feat=scaler.fit_transform(newfeats)
        atom_feats=np.concatenate([atom_feats,newfeats],axis=1)

        for i in range(0, mol.GetNumAtoms()):
            for j in range(0, mol.GetNumAtoms()):
                if adj_mat[i, j] == 1:
                    bonds.append([i, j])

        # for making I
        for i in range(mol.GetNumAtoms()):
            Identity.append([i,i])

        if len(bonds) == 0:
            return None

        atom_feats = torch.tensor(atom_feats, dtype=torch.float)
        bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
        Identity = torch.tensor(Identity, dtype=torch.long).t().contiguous()
        y = torch.tensor(target, dtype=torch.float).view(1, 1)

        extra_feature = extract_IntrinsicF(mol, feature_num)
        
        # print(Identity)
        # print(bonds)
        # print(f'Adj Size {adj_mat.shape}, #atom {mol.GetNumAtoms()}, Id {Identity.shape}, bonds {bonds.shape}')
        return Data(x=atom_feats,y=y, edge_index=bonds, idx=idx, eFeature = extra_feature, id_index=Identity, smiles=smiles)
                    # MW=MW, LOGP=LOGP, HBA=HBA, HBD=HBD, rotable=rotable, amide=amide, bridge=bridge, heteroA=heteroA, heavy=heavy, spiro=spiro, FCSP3=FCSP3, ring=ring, Aliphatic=Aliphatic, aromatic=aromatic, saturated=saturated, heteroR=heteroR, TPSA=TPSA, valence=valence, mr=mr)
    except:

        print("err")
        return 

def smiles_to_ECFP(elem_feats, smiles, idx, target):
    try:
        mol = Chem.MolFromSmiles(smiles)
        adj_mat = Chem.GetAdjacencyMatrix(mol)
        # adding Identity Matrix 
        # adj_mat = adj_mat + numpy.eye(len(adj_mat)) 
        atom_feats = list()
        bonds = list()
        newfeats=list()
        for atom in mol.GetAtoms():
            atom_feats.append(elem_feats[atom.GetAtomicNum() - 1, :])
            newfeats.append([atom.GetTotalNumHs(),atom.IsInRing()*1,atom.GetTotalDegree(),
                             atom.GetTotalValence(),atom.GetIsAromatic()*1])
        
        atom_feats=np.concatenate([atom_feats,newfeats],axis=1)
        
        for i in range(0, mol.GetNumAtoms()):
            for j in range(0, mol.GetNumAtoms()):
                if adj_mat[i, j] == 1:
                    bonds.append([i, j])

        if len(bonds) == 0:
            return None

        atom_feats = torch.tensor(atom_feats, dtype=torch.float)
        bonds = torch.tensor(bonds, dtype=torch.long).t().contiguous()
        y = torch.tensor(target, dtype=torch.float).view(1, 1)
        
        
        extra_feature = extract_IntrinsicF(mol)
        
        radius=3
        nBits=1024
        ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol,radius=radius, nBits=nBits)
        ECFP = torch.tensor(np.float32(ECFP6)).view(1,-1)

        Mold2 = []
        return Data(x=atom_feats, y=y, edge_index=bonds, ecfp=ECFP,Mold2=Mold2, idx=idx, eFeature = extra_feature)
                    # MW=MW, LOGP=LOGP, HBA=HBA, HBD=HBD, rotable=rotable, amide=amide, bridge=bridge, heteroA=heteroA, heavy=heavy, spiro=spiro, FCSP3=FCSP3, ring=ring, Aliphatic=Aliphatic, aromatic=aromatic, saturated=saturated, heteroR=heteroR, TPSA=TPSA, valence=valence, mr=mr, info=hi)
    except:

        return

# Intrinsic property 
def extract_IntrinsicF(mol, feature_num=20):
    import rdkit 
    ## 0. from eGCN-2
    if feature_num == 0:
        return

    if feature_num == 2 :
        MolWt = rdkit.Chem.Descriptors.MolWt(mol)
        RingCount = rdkit.Chem.Lipinski.RingCount(mol)
        recom = [MolWt, RingCount]
    
    else :
        ## 1. Highlight Feature from CCEL 
        ## 20 EA 
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

    if feature_num == 20:
        recom = [MolWt, HeavyAtomMolWt, NumValenceElectrons, FractionCSP3, HeavyAtomCount, NHOHCount, NOCount, NumAliphaticCarbocycles, NumAliphaticHeterocycles, NumAliphaticRings, NumAromaticCarbocycles, NumAromaticHeterocycles, NumAromaticRings, NumHAcceptors, NumHDonors, NumHeteroatoms, NumRotatableBonds, RingCount, MolMR, CalcNumBridgeheadAtom]
    elif feature_num == 32 :
        notRecom = [ExactMolWt, NumRadicalElectrons, MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge, MinAbsPartialCharge, NumSaturatedCarbocycles, NumSaturatedHeterocycles, NumSaturatedRings, MolLogP, CalcNumAmideBonds, CalcNumSpiroAtoms]
        recom += notRecom
    
    return torch.tensor(recom).view(1,-1)


if __name__ == '__main__':
    print("아녕!")

    target1 =  Chem.MolFromSmiles('CCCC1=CCCC1')
    target2 =  Chem.MolFromSmiles('CCCC1=CCCC1')

    print(target1)
    print(target2)