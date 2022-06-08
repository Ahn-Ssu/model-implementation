import os
import pandas as pd

savePath = "/home/ahn_ssu/CP2GN2/5_predict/2021-09-16"
savePath += "/"

files = os.listdir(savePath)

exp_label = "SAGE model_ wo_B | ep4.5__"

count = 0
for idx, one in enumerate(files):
    print(f'now idx:{idx}, file name:{one}, count:{count}', end="\r")
    # if not idx:
    data = pd.read_excel(savePath+one,header=None)
    count += data.shape[0]
    
    # else:
        # data = pd.concat([data, pd.read_excel(savePath+one,header=None)],axis=1)

print() # for end processing

# data.dropna()
# print(data.shape)

# data.to_excel(savePath + exp_label + "total.xlsx", index=False)