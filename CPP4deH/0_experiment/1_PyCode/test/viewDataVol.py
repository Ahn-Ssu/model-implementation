import utils.ml
import utils.chem as chem
import argparse
import pandas as pd
from tqdm import tqdm

# Setting
paser = argparse.ArgumentParser()
args = paser.parse_args("")

args.input_dim = utils.chem.n_atom_feats
args.ex_dim = 20

# Load datase
print('Load molecular structures...')
dataPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/2_CCELnQM9.xlsx"
dataSet = chem.load_dataset(dataPath, featrue_num= args.ex_dim)

forSave = []

print("Extracting")
for one in tqdm(dataSet):
    print(f'SMILES {one[0]}\n\t edge_index={one[1].edge_index.shape[1]}, x={one[1].x.shape[0]}, y={one[1].y[0][0]}')
    forSave.append([one[0], one[1].edge_index, one[1].edge_index.shape[1], one[1].x ,one[1].x.shape[0], float(one[1].y[0][0])])



print("Saving")
df = pd.DataFrame(forSave)
# edge_index는 graph 안에 total edge가 얼마냐 있느냐를 나타내는 것
# x는 atom마다 붙게되는데, 결국엔 분자의 크기 자체는 atom의 숫자여야 함
df.columns = ['SMILES','edge_index','edge Size','x(feature)','x Size', 'deH']
df.to_excel('/home/ahn_ssu/CP2GN2/2_data/1_DataSet/forDualGNN.xlsx',index=False)

print("\tCode Run Done")
print("\tCode Run Done")
