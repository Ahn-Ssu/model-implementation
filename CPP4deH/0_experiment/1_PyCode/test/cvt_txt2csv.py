import os, csv
from tqdm import tqdm


filePath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/0_Origin dataSet archive/prediction_smi.txt"
f = open(filePath, 'r')



all = f.readlines()
f.close()

targetPath = "/home/ahn_ssu/CP2GN2/2_data/1_DataSet/"
targetPath += "prediction_smi.csv"
if not os.path.exists(targetPath):
      with open(targetPath, mode='w') as f:
        myWriter = csv.writer(f)
        # myWriter.writerow(["stage","epoch","MAE","Std","R2"])


for one in tqdm(all):
    # print(one)
    S, val = one.split()
    with open(targetPath, mode='a') as f:  
        myWriter = csv.writer(f)
        myWriter.writerow([S, val])
    # exit()


