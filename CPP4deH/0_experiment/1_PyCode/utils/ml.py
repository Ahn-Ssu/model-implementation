import sys
sys.path.append("/home/ahn_ssu/CP2GN2/0_experiment/1_PyCode/utils/")

import time
import math
import copy
import torch
import numpy as np

from tqdm import tqdm
from torch_geometric.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from model.models import *

def GPU_check():
    N_GPU = torch.cuda.device_count()
    idx_str = ""

    if N_GPU == 0 :
        return "cpu"
    
    for idx in range(N_GPU):

        idx_str += str(idx) + " "
        print(
        """        -------------------- GPU Info. --------------------
            GPU idx : name = : {} [{}]
            cuda (max, min) capability : {}
            Allocated : {} GB\tCached : {} GB
        ---------------------------------------------------""".format(
            idx,
            torch.cuda.get_device_name(idx),
            torch.cuda.get_device_capability(idx),
            torch.cuda.max_memory_allocated(idx),
            torch.cuda.memory_reserved(idx))) # memory_cached -> memory_reserved

    if N_GPU == 1 :
        return GPU_set(0)

    print("available :",idx_str)
    request_idx = int(input(">> enter GPU idx : "))
    return GPU_set(request_idx)



def GPU_set(GPU_idx):
    
    MODEL_NAME = 'GNN' 
    DEVICE = torch.device(f'cuda:{GPU_idx}' if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(DEVICE)
    print(
        """ 
        -------------------- GPU Info. --------------------
            Request GPU idx : {}
            CUDA Available : {}
            CUDA DEVICE totalNum : {}
            CUDA Current DEVICE Index : Name = {} : [{}]
            CUDA DEVICE Capability(version) : {}

            [Memory]
            Allocated : {} GB
            Cached : {} GB
        ---------------------------------------------------
        """.format(GPU_idx,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        torch.cuda.current_device(),
        torch.cuda.get_device_name(torch.cuda.current_device()),
        torch.cuda.get_device_capability(GPU_idx),
        round(torch.cuda.memory_allocated(GPU_idx), 1),
        round(torch.cuda.memory_reserved(GPU_idx),1)))

    return DEVICE

# utility function to measure time
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

def train(model, optimizer, data_loader, criterion,DEVICE):
    model.train()
    train_loss = 0

    for i, (batch) in enumerate(data_loader):
        
        pred = model(batch.to(DEVICE))
        loss = criterion(pred, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(data_loader)


def validate(model, data_loader, criterion,DEVICE):
  model.eval()
  val_loss =0
  
  with torch.no_grad():
    for i, (batch) in enumerate(data_loader):
        preds = model(batch.to(DEVICE))
        loss = criterion(preds, batch.y)
        val_loss += loss.item()

    val_loss = val_loss / len(data_loader)
  
  return val_loss

def test(model, data_loader,DEVICE):
    model.eval()
    list_preds = list()
    
    with torch.no_grad():
        for batch in data_loader:
            preds = model(batch.to(DEVICE))
            list_preds.append(preds)

    return torch.cat(list_preds, dim=0).cpu().numpy()



def crossValidation(
                data = None,
                test1_data = None,
                # test2_data = None,
                n_splits = 5, random_state=400, args=None, savePath=".", case=""):
    import os
    import csv 
    from sklearn.model_selection import KFold
    from datetime import date, datetime, timezone, timedelta


    kfold = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    exp_day = str(date.today())
    savePath = savePath + "/" + exp_day

    KST = timezone(timedelta(hours=9))
    time_record = str(datetime.now(KST).time())[:8]

    start = time.time()

    # log_ = savePath + "/" + time_record + case+"_Train_log.csv"
    evaluation_ = savePath + "/" + time_record + case + "_evaluation_log.csv"
    extra = savePath + "/"

    try:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
    except OSError:
        print("Error: Cannot create the directory {}".format(savePath))

    # if not os.path.exists(log_): # 파일이 없는 경우 생성하고, Label을 우선 붙임 
    #   with open(log_, mode='w') as f:
    #     myWriter = csv.writer(f)
    #     myWriter.writerow(["stage","epoch","loss"])

    if not os.path.exists(evaluation_):
      with open(evaluation_, mode='w') as f:
        myWriter = csv.writer(f)
        # myWriter.writerow(["stage","epoch","MAE","RMSE","R2"])
        myWriter.writerow(["stage","epoch","vMAE","vR2","tMAE","tR2"])
        # myWriter.writerow(["stage","epoch","vRMSE","vR2","tRMSE","tR2"])
        # myWriter.writerow(["stage","epoch","accuracy_score", "balanced_accuracy_score", "recall_score", "precision_score", "f1_score", "matthews_corrcoef", "roc_auc_score", "average_precision_score", "precision_recall_curve"])
        
    print("""\t\t---------- parameter seting ---------\n\t{}""".format(str(args)))

    eachTrain_time = list()

    for stage, (train_index, validate_index) in enumerate(kfold.split(data)): #cross validation, k=5

        # if stage < 4:
        #     continue
        
        print("""

        --- now validation Round : {} ---""".format(stage+1))

        train_data=list()
        validate_data=list()
        # test1_data=list()
        # test2_data=list()
        
        for i in train_index:
            train_data.append(data[i])

        
        for i in validate_index:
            validate_data.append(data[i])
        
        # for i in range(len(test1)):
        #     test1_data.append(test1[i])
        
        # for i in range(len(test2)):
        #     test2_data.append(test2[i])

        # MAE 모니터링을 위한 Extract 
        valid_targets=np.array([x.y.item() for x in validate_data]).reshape(-1, 1)
        train_targets=np.array([x.y.item() for x in train_data]).reshape(-1, 1)
        test1_targets=np.array([x.y.item() for x in test1_data]).reshape(-1, 1)
        # test2_targets=np.array([x.y.item() for x in test2_data]).reshape(-1, 1)
        # for data set 
        valid_smiles=np.array([x.smiles for x in validate_data]).reshape(-1, 1)
        train_smiles=np.array([x.smiles for x in train_data]).reshape(-1, 1)
        test1_smiles=np.array([x.smiles for x in test1_data]).reshape(-1, 1)

        # v = np.concatenate([valid_smiles,valid_targets], axis=1)
        # t = np.concatenate([train_smiles,train_targets], axis=1)

        
        with open(extra+"BP_val"+str(stage)+".csv", mode='w') as f:
                myWriter = csv.writer(f)
                # myWriter.writerow(["stage","epoch","MAE","RMSE","R2"])
                for i in range(len(valid_smiles)):
                    myWriter.writerow([valid_smiles[i].item(),valid_targets[i].item()])

        
        with open(extra+"BP_test.csv", mode='w') as f:
                myWriter = csv.writer(f)
                # myWriter.writerow(["stage","epoch","MAE","RMSE","R2"])
                for i in range(len(test1_smiles)):
                    myWriter.writerow([test1_smiles[i].item(),test1_smiles[i].item()])

        continue
                
        
        # train_data, validate_data, test1_data = m_feat_normalize(train= train_data, valid=validate_data, test1=test1_data)#, test2=test2_data)

        # KFold에 의해 나누어진 코드들을 학습에 사용하기 위해서 데이터 셋으로 변환 
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(validate_data, batch_size=100)
        test1_loader = DataLoader(test1_data, batch_size=100)
        # test2_loader = DataLoader(test2_data, batch_size=100)

        # 모델 초기화 
        model = args.model(args).to(args.DEVICE)
        if not stage:
            print(model)
        optimizer = args.optimizer(model.parameters(), lr=args.init_lr, weight_decay=args.l2_coeff)
        criterion = args.criterion()
        # criterion = torch.nn.MSELoss()

        stage_indicator = "Round : " + str(stage+1) + " | Epoch"
        print("\n\n")
        print(case)

        trainTime = time.time()
        for idx in tqdm(range(0, args.n_epochs), desc=stage_indicator): #각 cross validation 을 지정된 epoch으로 실행  
            train(model, optimizer, train_loader, criterion, args.DEVICE)

            # with open(log_, mode='a') as f:  
            #   myWriter = csv.writer(f)
            #   myWriter.writerow([stage+1,idx+1,train_loss])

            if (idx+1) % 1 == 0 :
              preds_v = test(model, valid_loader, args.DEVICE)
              v_mae = mean_absolute_error(valid_targets, preds_v)
            #   v_rmse = mean_squared_error(valid_targets, preds_v)**0.5
              v_r2 = r2_score(valid_targets, preds_v)

              preds_t1 = test(model, test1_loader, args.DEVICE)
              t1_mae = mean_absolute_error(test1_targets, preds_t1)
            #   t1_rmse = mean_squared_error(test1_targets, preds_t1)**0.5
              t1_r2 = r2_score(test1_targets, preds_t1)

            #   preds_t2 = test(model, test2_loader, args.DEVICE)
            #   t2_mae = mean_absolute_error(test2_targets, preds_t2)
            #   t2_r2 = r2_score(test2_targets, preds_t2)

            #   preds = np.rint(preds)
            #   acc = accuracy_score(valid_targets, preds)
            #   b_acc = balanced_accuracy_score(valid_targets, preds)
            #   recall = recall_score(valid_targets, preds)
            #   precision = precision_score(valid_targets, preds)
            #   f1 = f1_score(valid_targets, preds)
            #   mat_ccoef =  matthews_corrcoef(valid_targets, preds)
            #   roc_auc = roc_auc_score(valid_targets, preds)
            #   aps = average_precision_score(valid_targets, preds)
            #   prc = precision_recall_curve(valid_targets, preds)
              

            #   preds = np.rint(preds)
            #   acc = accuracy_score(valid_targets, preds)
            #   b_acc = balanced_accuracy_score(valid_targets, preds)
            #   recall = recall_score(valid_targets, preds)
            #   precision = precision_score(valid_targets, preds)
            #   f1 = f1_score(valid_targets, preds)
            #   mat_ccoef =  matthews_corrcoef(valid_targets, preds)
            #   roc_auc = roc_auc_score(valid_targets, preds)
            #   aps = average_precision_score(valid_targets, preds)
            #   prc = precision_recall_curve(valid_targets, preds)
              

              with open(evaluation_, mode='a') as f:  
                myWriter = csv.writer(f)
                # myWriter.writerow([stage+1, idx+1, test_mae, test_mse, r2])
                myWriter.writerow([stage+1, idx+1, v_mae, v_r2, t1_mae, t1_r2])#, t2_mae, t2_r2])
                # myWriter.writerow([stage+1, idx+1, v_rmse, v_r2, t1_rmse, t1_r2])#, t2_mae, t2_r2])
                # myWriter.writerow([stage+1, idx+1, acc, b_acc, recall, precision, f1, mat_ccoef, roc_auc, aps, prc])

        eachTrain_time.append(timeSince(trainTime))
        # preds = test(model, valid_loader, args.DEVICE)
        # valid_targets=np.array([x.y.item() for x in validate_data]).reshape(-1, 1)
        # tmp_mae = np.mean(np.abs(valid_targets - preds))
        # print(i+1,' validation mae: ',tmp_mae)
        # valid_mae = valid_mae+tmp_mae
    
    # with open(evaluation_, mode='a') as f:  
    #             myWriter = csv.writer(f)
    #             myWriter.writerow(eachTrain_time)
    #             myWriter.writerow(["Time Take",timeSince(start)])


import copy

def a_feat_normalize(train,valid,test1
#,test2=None
        ):
       
    train_dt1=train[0]['x']
    valid_dt1=valid[0]['x']
    test1_dt1=test1[0]['x']
    # test2_dt1=test2[0]['x']

    for i in range(1,len(train)):
        train_dt2=train[i]['x']
        train_dt1=np.concatenate((train_dt1,train_dt2))    
    for i in range(1,len(valid)):    
        valid_dt2=valid[i]['x']
        valid_dt1=np.concatenate((valid_dt1,valid_dt2))        
    for i in range(1,len(test1)):    
        test1_dt2=test1[i]['x']
        test1_dt1=np.concatenate((test1_dt1,test1_dt2))
    # for i in range(1,len(test2)):    
    #     test2_dt2=test2[i]['x']
    #     test2_dt1=np.concatenate((test2_dt1,test2_dt2))
    
    
    scaler = MinMaxScaler()
    n_train = scaler.fit_transform(train_dt1)
    n_valid = scaler.transform(valid_dt1)
    n_test1 = scaler.transform(test1_dt1)
    # n_test2 = scaler.transform(test2_dt1)

    cur_len = 0
    for i in range(0,len(train)):
        len_atom = len(train[i]['x'])
        train[i]['x'] = torch.tensor(n_train[range(cur_len,cur_len+len_atom)])
        cur_len = cur_len+len_atom

    cur_len = 0
    for i in range(0,len(valid)):
        len_atom = len(valid[i]['x'])
        valid[i]['x'] = torch.tensor(n_valid[range(cur_len,cur_len+len_atom)])
        cur_len = cur_len+len_atom
        
    cur_len = 0
    for i in range(0,len(test1)):
        len_atom = len(test1[i]['x'])
        test1[i]['x'] = torch.tensor(n_test1[range(cur_len,cur_len+len_atom)])
        cur_len = cur_len+len_atom

    # cur_len = 0
    # for i in range(0,len(test2)):
    #     len_atom = len(test2[i]['x'])
    #     test2[i]['x'] = torch.tensor(n_test2[range(cur_len,cur_len+len_atom)])
    #     cur_len = cur_len+len_atom

    return copy.deepcopy(train),copy.deepcopy(valid),copy.deepcopy(test1) #, copy.deepcopy(test2) 

def m_feat_normalize(train,valid,test1
        #, test2
        ):
       
    train_dt1=train[0]['eFeature']
    valid_dt1=valid[0]['eFeature']
    test1_dt1=test1[0]['eFeature']
    # test2_dt1=test2[0]['eFeature']

    for i in range(1,len(train)):
        # print(f"now idx: {i}")
        train_dt2=train[i]['eFeature']
        # print(train_dt2)
        train_dt1=np.concatenate((train_dt1,train_dt2))    
        # print("after", train_dt1)
        # print(train_dt1.shape)
        # input()
    for i in range(1,len(valid)):    
        valid_dt2=valid[i]['eFeature']
        valid_dt1=np.concatenate((valid_dt1,valid_dt2))
    for i in range(1,len(test1)):    
        test1_dt2=test1[i]['eFeature']
        test1_dt1=np.concatenate((test1_dt1,test1_dt2))   
    # for i in range(1,len(test2)):    
    #     test2_dt2=test2[i]['eFeature']
    #     test2_dt1=np.concatenate((test2_dt1,test2_dt2))        
    import pandas as pd 

    # print(train_dt1)
    # print(test1_dt1)

    # train_df = pd.DataFrame(train_dt1)
    # test_df = pd.DataFrame(test1_dt1)
    # train_df.to_excel( "/home/ahn_ssu/CP2GN2/TRAIN_Ef.xlsx")
    # test_df.to_excel( "/home/ahn_ssu/CP2GN2/Test_Ef.xlsx")

    scaler = MinMaxScaler()
    n_train = scaler.fit_transform(train_dt1)
    n_valid = scaler.transform(valid_dt1)
    n_test1 = scaler.transform(test1_dt1)
    # n_test2 = scaler.transform(test2_dt1)

    # print(n_train)
    # print()
    # input()
    # print(n_train[1])
    # print()
    # input()
    # print(n_test1)
    # print()
    # input()
    # print(n_test1[1])
    # print()
    # input()

    # train_df = pd.DataFrame(n_train)
    # test_df = pd.DataFrame(n_test1)
    # train_df.to_excel( "/home/ahn_ssu/CP2GN2/normTRAIN_Ef.xlsx")
    # test_df.to_excel( "/home/ahn_ssu/CP2GN2/normTest_Ef.xlsx")
    
    for i in range(0,len(train)):
        temp = torch.tensor([n_train[i]])    
        # print(temp)
        # print(temp.shape)
        # input()
        train[i]['eFeature'] = temp
    for i in range(0,len(valid)):
        valid[i]['eFeature'] = torch.tensor([n_valid[i]])
    for i in range(0,len(test1)):    
        test1[i]['eFeature'] = torch.tensor([n_test1[i]])  

        temp = torch.tensor([n_test1[i]])  
        # print(temp)
        # print(temp.shape)
        # input()
    # for i in range(0,len(test2)):    
    #     test2[i]['eFeature'] = torch.tensor([n_test2[i]])

    return copy.deepcopy(train),copy.deepcopy(valid),copy.deepcopy(test1) #,copy.deepcopy(test2)


if __name__ == '__main__':
    print("Test GPU check & set Result__", GPU_check())
