# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 23:45:29 2025

@author: Chalermwong
"""

import pandas as pd
import sys
from fn_charmy import *
import numpy as np
import os

path = 'C:/Users/Chalermwong/OneDrive/Desktop/bot/'

# df_reward = pd.read_csv(path+'rewardResult.csv')
# df_indicator = pd.read_csv(path+'indicatorResult.csv')


###############################################################################

# SEQ_LEN = 60   # sequence length กำหนดเอง
# BATCH_SIZE = 1024 #512 1024
# actionWhen = 2


###############################################################################
df_reward = pd.DataFrame()
folder_path = 'datareward'

for filename in os.listdir(folder_path):
    df_reward = pd.concat([df_reward,pd.read_csv(folder_path+'/'+filename,low_memory=False)])

df_reward = df_reward.drop_duplicates('time')
df_reward = df_reward.reset_index(drop=1)
df_reward = df_reward.sort_values('time').reset_index(drop=1)
###############################################################################
df_indicator = pd.DataFrame()
folder_path = 'dataindicator'

for filename in os.listdir(folder_path):
    df_indicator = pd.concat([df_indicator,pd.read_csv(folder_path+'/'+filename,low_memory=False)])

df_indicator = df_indicator.drop_duplicates('time')
df_indicator = df_indicator.reset_index(drop=1)
df_indicator = df_indicator.sort_values('time').reset_index(drop=1)
###############################################################################

df_columns = pd.DataFrame(df_indicator.columns)
df_columns['Col'] = df_columns[0]
df_columns[['tf_text','indi']] = df_columns['Col'].str.split(':', expand=True )
df_columns = df_columns.loc[df_columns['tf_text'].isin(df_dataLoad['tf_text'])].reset_index(drop=1)

df_columns2 = df_columns.loc[df_columns['indi']=='ratioValue'].reset_index(drop=1)

rate = 0.2

for i in range (df_columns2.shape[0]):
    df_indicator[df_columns2['Col'][i]].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    
    df_indicator['tmp'] = df_indicator[df_columns2['Col'][i]].copy()
     
    df_indicator.loc[:, df_columns2['Col'][i]] =  (-df_indicator[df_columns2['Col'][i]].abs()//rate)*rate
    
    df_indicator.loc[df_indicator['tmp']>=0, df_columns2['Col'][i]] = df_indicator[df_columns2['Col'][i]] *-1

del df_indicator['tmp']


# shift value to increse col
# for i in range (df_columns.shape[0]):
#     for shift in range (1,6):
#         df_indicator.loc[:, df_columns['Col'][i]+':'+str(shift)] = df_indicator[df_columns['Col'][i]].shift(shift)
    
    
mapping = {
    True: 1,
    'up': 1,
    'down': -1,
    '-':0,
    np.nan: 0,
    'buy':1,
    'sell':-1
}

###############################################################################


def optimize_df(df, use_category=True):
    optimized = df.copy()
    start_mem = optimized.memory_usage(deep=True).sum() / 1024**2
    report = []

    for col in optimized.columns:
        col_type = optimized[col].dtype
        col_mem_before = optimized[col].memory_usage(deep=True) / 1024**2
        new_type = col_type

        if pd.api.types.is_float_dtype(col_type):
            # เช็คว่าทุกค่าลงตัวเป็น int ได้มั้ย
            if np.allclose(optimized[col].dropna() % 1, 0):
                optimized[col] = pd.to_numeric(optimized[col], downcast="integer")
                new_type = optimized[col].dtype
            else:
                optimized[col] = pd.to_numeric(optimized[col], downcast="float")
                new_type = optimized[col].dtype

        elif pd.api.types.is_integer_dtype(col_type):
            optimized[col] = pd.to_numeric(optimized[col], downcast="integer")
            new_type = optimized[col].dtype

        elif pd.api.types.is_object_dtype(col_type):
            try:
                # ลองแปลงเป็น numeric
                tmp = pd.to_numeric(optimized[col], errors="raise")
                if np.allclose(tmp.dropna() % 1, 0):
                    optimized[col] = pd.to_numeric(tmp, downcast="integer")
                else:
                    optimized[col] = pd.to_numeric(tmp, downcast="float")
                new_type = optimized[col].dtype
            except:
                if use_category:
                    optimized[col] = optimized[col].astype("category")
                    new_type = "category"

        col_mem_after = optimized[col].memory_usage(deep=True) / 1024**2
        report.append([col, col_type, new_type, col_mem_before, col_mem_after])

    end_mem = optimized.memory_usage(deep=True).sum() / 1024**2

    # สรุปผล
    report_df = pd.DataFrame(report, columns=["Column", "Before", "After", "MemBefore(MB)", "MemAfter(MB)"])
    print(f"\nMemory usage reduced: {start_mem:.2f} MB → {end_mem:.2f} MB "
          f"({100*(start_mem-end_mem)/start_mem:.1f}% saved)")
    return optimized, report_df

###############################################################################

#%%

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ======================
# 1. Dataset (รองรับ data ใหญ่)
# ======================
# SEQ_LEN = 60   # sequence length กำหนดเอง
# BATCH_SIZE = 512 #512 1024

class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=150):
        self.seq_len = seq_len
        self.features = df.drop(columns=["action"]).values.astype(np.float32)
        self.labels = df["action"].values + 1
        self.indices = []  # เก็บ mapping กลับไปยัง df index

        self.sequences, self.seq_labels = [], []
        for i in range(len(self.features) - seq_len):
            self.sequences.append(self.features[i:i+seq_len])
            self.seq_labels.append(self.labels[i+seq_len-1])
            self.indices.append(i+seq_len-1)  # index ของ label ใน df

        self.sequences = torch.tensor(self.sequences, dtype=torch.float32)
        self.seq_labels = torch.tensor(self.seq_labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.seq_labels[idx], self.indices[idx]

# ======================
# 2. Models
# ======================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out,_ = self.lstm(x)                 # (batch, seq, hidden)
        out = out[:, -1, :]                  # last timestep
        out = self.bn(out)
        return self.fc(out)

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out,_ = self.gru(x)
        out = out[:, -1, :]
        out = self.bn(out)
        return self.fc(out)

# ======================
# 3. Training Function (with Early Stopping)
# ======================
def train_model(model, dataloader, num_epochs=50, lr=0.005, patience=5, min_loss = 0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch, idx_batch  in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"{model.__class__.__name__} Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # ✅ ถ้า loss ต่ำกว่า threshold หยุดเลย
        if avg_loss <= min_loss:
            print(f"Stopping early: Loss reached {avg_loss:.4f} <= {min_loss}")
            break
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model

# ======================
# 4. Evaluation Function
# ======================
def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch, idx_batch  in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["-1","0","1"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["-1","0","1"], yticklabels=["-1","0","1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({model.__class__.__name__})")
    plt.show()

def predict_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch, idx_batch  in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            # y_true.extend(y_batch.numpy())
            y_pred.extend(preds)

    return y_pred

#%%


###############################################################################


    
df_indicator = df_indicator.replace(mapping)   

df_indicatorHead = df_indicator.iloc[:,:3].reset_index(drop=1)
df_indicator = df_indicator.iloc[:,3:].reset_index(drop=1)

df_indicator, report_optimize_df = optimize_df(df_indicator)

df_indicator = pd.concat([df_indicatorHead,df_indicator],axis=1)

###############################################################################
    


df_reward2 = df_reward.copy()
df_reward2 = df_reward2[['time','3:act','5:act', '10:act', '15:act', '30:act', '60:act', '120:act', '240:act','480:act', '960:act', '1200:act', '1500:act']]
df_reward2 = df_reward2.replace(mapping)  
# df_reward2['sum'] = df_reward2['3:act'] + df_reward2['5:act'] + df_reward2['10:act'] + df_reward2['15:act'] + df_reward2['30:act'] 
df_reward2['sum'] =  df_reward2['15:act'] + df_reward2['30:act'] + df_reward2['60:act'] + df_reward2['120:act'] 

###############################################################################


for actionWhen in [1]:
    

    df_reward2.loc[df_reward2['sum']>=actionWhen,'action'] = 1 
    df_reward2.loc[df_reward2['sum']<=-actionWhen,'action'] = -1 
    df_reward2.loc[df_reward2['action'].isna(), 'action'] = 0 
    
    ###############################################################################    
    
    
    ###############################################################################
    df_formodel = pd.merge(df_indicator,df_reward2[['time','action']],on='time',how='left')    
    df_formodel = df_formodel.dropna()
    df_formodelHead = df_formodel.iloc[:,:3].reset_index(drop=1)
    df_formodel = df_formodel.iloc[:,3:].reset_index(drop=1)
    ###############################################################################
    
    for SEQ_LEN in [20]: #[5,10,15,20,30]
        for BATCH_SIZE in [128,256]: #[64,128,256,512,789] 128,256,512
    
            print (actionWhen,'-', str(SEQ_LEN),'-',str(BATCH_SIZE))
    
    
    
            # df_formodel, report_optimize_df = optimize_df(df_formodel)
            
            #############################################################################################################
            
            
            
            
            # df = pd.DataFrame(data)
            
            # dataset = TimeSeriesDataset(df, seq_len=SEQ_LEN)
            # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            
            
            # '2014-10-13 09:23:00'
            rangeTrain = df_formodelHead.loc[(df_formodelHead['time_convert']>='2023-01-01') & (df_formodelHead['time_convert']<='2024-12-15')]
            
            rangeTest = df_formodelHead.loc[(df_formodelHead['time_convert']>='2025-01-01')]
            
            
            
            df_train = df_formodel.iloc[rangeTrain.index.min():rangeTrain.index.max()]
            dataset_train = TimeSeriesDataset(df_train, seq_len=SEQ_LEN)
            dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
            
            
            df_test = df_formodel.iloc[rangeTest.index.min():rangeTest.index.max()]
            dataset_test = TimeSeriesDataset(df_test, seq_len=SEQ_LEN)
            dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
            
            
            
            input_size = df_train.shape[1] - 1
            models = {'LSTM':LSTMClassifier(input_size),
                'GRU': GRUClassifier(input_size),
            }
            
            
            for idx, model in enumerate( models.values() ):
                
                trained_model = train_model(model, dataloader, num_epochs=100)
                evaluate_model(trained_model, dataloader)
                
                torch.save(model.state_dict(), 'model2/'+list(models.keys())[idx]+'_actionWhen-'+str(actionWhen)+'_SEQ_LEN-'+str(SEQ_LEN)+'_BATCH_SIZE-'+str(BATCH_SIZE)+'.pth')
            
            # X, y, idx = dataset_train[0]
            # print("Sequence shape:", X.shape)
            # print("Label:", y)
            # print("Original df index:", idx)
            # print("Original df row:\n", df_train.iloc[idx])
            
            
            # X, y, idx = dataset_train[0]
            # X, y, idx = dataset_test[0]
            # idx+rangeTest.index.min()
            
            
            
            
            result ={}
            for idx, model in enumerate( models.values() ):
                # print (idx, model)
                y_pred = predict_model(model, dataloader_test)
                y_pred = np.array([int(x)-1 for x in y_pred])
            
                # evaluate_model(model, dataloader_test)
                result[list(models.keys())[idx]] = y_pred
                
            for i in ['time', 'time_convert']:
                result[i] = df_formodelHead.iloc[SEQ_LEN + rangeTest.index.min()  : rangeTest.index.max()][i]
            
            dfResults = pd.DataFrame(result)
            dfResults = dfResults.sort_values('time').reset_index(drop=1)
            
            
            typeResample = 'dataTrain2023-2024'
            # a = 'xx-month'
            
            
            
            dataTrain = str(SEQ_LEN)+'-'+str(BATCH_SIZE)
            
            
            file = 'datapredict/df_predict_Deepshuffle=False_actionwhen'+str(actionWhen)+'_'+dataTrain+'_'+typeResample+'.csv'
            
            dfResults.to_csv(file,index=0)    
            
            
            
            
            # y_pred = predict_model(models.values[, dataloader_test)
            # evaluate_model(models[0], dataloader_test)
            # print("Unique labels:", dataset_train.seq_labels.unique())
            # print("Num classes in model:", num_classes)






























































