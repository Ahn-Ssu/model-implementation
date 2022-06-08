import deepchem 



task_label, diskDataset, transformer = deepchem.deepchem.molnet.load_lipo()

print(task_label)
print(diskDataset)
print(transformer)




print()
print("dataset indexing")
for idx, one in enumerate(diskDataset):
    print(idx, one)
    print()
print("dataset indexing----end")




train = diskDataset[1].to_dataframe()

print("train:\n",train)
extracting_list = ['ids','y']
edit_train = train[extracting_list]

print("edit_train:\n",edit_train)



