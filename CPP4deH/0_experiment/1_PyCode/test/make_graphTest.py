import networkx as nx 
import numpy as np
import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
from tqdm import tqdm

# path = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/0_DehydroH_new_dataSet_fromCCEL.xlsx"
path  = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/1_[processed]QM9_deH_SMILES.xlsx"
id_target = np.array(pd.read_excel(path))

smiles = id_target[:, 0]


for aSMILES in tqdm(smiles):
    mol = Chem.MolFromSmiles(aSMILES)
    adj = Chem.GetAdjacencyMatrix(mol)

    G = nx.Graph()

    for idx, neibor in enumerate(adj):
        G.add_node(idx, visit=[], label=aSMILES[idx])

    for idx, neibor in enumerate(adj):
      for i, one in enumerate(neibor):
            if one:
                G.add_edge(idx, i)

    #  idx를 레이블로 잡아서 하는 법 
    # nx.draw(G, with_labels=True)

    # adj로 매우 단순하게 보는 법 
    # result = nx.Graph(adj) 
    # nx.draw(result)  
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=400, node_color='skyblue', font_size=8, font_weight='bold', with_labels=True)
    # plt.tight_layout()
    plt.savefig("/home/ahn_ssu/CP2GN2/2_data/1_DataSet/img/1_QM9/"+aSMILES+"_Graph.png", format="PNG")
    plt.show(block=False)

    