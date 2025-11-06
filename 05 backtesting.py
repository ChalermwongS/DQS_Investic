# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:02:04 2025

@author: Chalermwong
"""

import pandas as pd
import os
import numpy as np
import sys

from multiprocessing import Pool,  cpu_count
import threading
from queue import Queue
from tqdm import tqdm

import time


dfRaw = {}
folder_path = 'data csv'
for filename in os.listdir(folder_path):
    dfRaw[filename.replace('.csv','')] = pd.read_csv(folder_path+'/'+filename)

df = dfRaw['1-m1'].copy()

# df_predict = pd.read_csv('df_predict sumcase2.csv')

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
                # if use_category:
                #     optimized[col] = optimized[col].astype("category")
                #     new_type = "category"
                pass

        col_mem_after = optimized[col].memory_usage(deep=True) / 1024**2
        report.append([col, col_type, new_type, col_mem_before, col_mem_after])

    end_mem = optimized.memory_usage(deep=True).sum() / 1024**2

    # สรุปผล
    report_df = pd.DataFrame(report, columns=["Column", "Before", "After", "MemBefore(MB)", "MemAfter(MB)"])
    print(f"\nMemory usage reduced: {start_mem:.2f} MB → {end_mem:.2f} MB "
          f"({100*(start_mem-end_mem)/start_mem:.1f}% saved)")
    return optimized, report_df

################################################################################

folderSummary = 'datasummary5'
folderPosition = 'dataposition5'

df_predict = pd.DataFrame()
folder_path = 'datapredict/run8'
for filename in os.listdir(folder_path):
    if not filename.endswith(".csv"):
        continue
    
    # ตัดชื่อไฟล์เป็น parts
    parts = filename.replace('.csv','').split('_')[2:] 
    # parts = ['setBlankto0', 'actionwhen2', '01-month', 'Under']
    
    tmp = pd.read_csv(os.path.join(folder_path, filename))
    tmpHead = tmp.iloc[:, -2:]  # เอา time, time_convert ไว้
    
    tmp_columns = tmp.columns
    
    # สร้าง suffix จาก parts (ยกเว้น 2 ตัวแรกถ้าไม่ต้องการ)
    # suffix = " : ".join(parts[:2])  # = "01-month : Under"
    suffix = parts[2] + " : " +parts[1] 
    
    # rename columns ที่ไม่ใช่ time
    for i in range(len(tmp_columns) - 2):
        tmp = tmp.rename(columns={tmp_columns[i]: tmp_columns[i] + " : " + suffix})
    
    # เก็บรวม
    if df_predict.empty:
        df_predict = tmp
    else:
        df_predict = pd.merge(df_predict, tmp, on=tmpHead.columns.tolist(), how="outer")


time_cols = ["time", "time_convert"]
other_cols = [c for c in df_predict.columns if c not in time_cols]
df_predict = df_predict[other_cols + time_cols]

df_predict = df_predict.dropna().reset_index(drop=1)

# df_predict = df_predict.loc[df_predict['time_convert']<='2025-02-05']

df_predict, report_optimize_df = optimize_df(df_predict)
###############################################################################

columnsListModel = df_predict.iloc[:,:-2].columns

dfuse = pd.merge(df,df_predict,how='left',on='time')

dfuse = dfuse.dropna().reset_index(drop=1)






dfuse, report_optimize_df = optimize_df(dfuse)
df, report_optimize_df = optimize_df(df)




#%%
df_condition = pd.read_excel('df_condition.xlsx', sheet_name='case2')
df_condition = df_condition.drop_duplicates()
df_condition = df_condition.reset_index(drop=1)
df_condition['note'] = df_condition['note'].astype(str)

df_condition = df_condition.loc[df_condition['exit reverse']==False].reset_index(drop=1)


df_noteModel = pd.read_csv('noteModel.csv')

df_condition = df_condition.loc[df_condition['note'].astype(int).isin(df_noteModel['noteModel'])].reset_index(drop=1)

df_condition = df_condition.loc[df_condition['note'].astype(int).isin([9683,12275,8014,8015,8069])]

# df_condition['note'] = '..'
#%%




# for idxCondition in list_idxCondition:
    
def backtest(idxCondition):    
    
    df_position = pd.DataFrame()
    
    # print ( str(idxCondition + 1 )+'/'+ str(df_condition.shape[0]), df_condition.iloc[idxCondition]['note']) 
    
    note = str(idxCondition) + ' ' +df_condition.iloc[idxCondition]['note']
    maxTP = df_condition.iloc[idxCondition]['tp']
    maxLoss = df_condition.iloc[idxCondition]['sl']
    maxHold = df_condition.iloc[idxCondition]['limit']
    entryPositionWhenCount = df_condition.iloc[idxCondition]['entryPositionWhenCount']
    
    
    # idx = 11
    # model = 'Combine : 01-month : Under : ahead'
    
    
    for idx, model in enumerate(columnsListModel):
        # print ( str(idx + 1 )+'/'+ str(len(columnsListModel)), model) 
        
        onHandPosition = 0
        status = {'position':99,
                  'count':0}
        entryPrice = 0
        entryTime = ''
        
        caseProtect = ''
        priceProtect = 0
        
        possibleLoss = 0
        
        if len(model[:2]) == 2: # 'RF':
        # if model == 'Combine : 01-month : Under : ahead black120': # 'RF':
            # print ( str(idx + 1 )+'/'+ str(len(columnsListModel)), model) 
            for i in range (dfuse.shape[0]):
            
                if dfuse.iloc[i][model] == status['position']:
                    status['count'] = status['count']+1
                else:
                    status['position'] = dfuse.iloc[i][model]
                    status['count'] = 1
                
                ## exit tp
                if df_condition.iloc[idxCondition]['exit tp'] == True:
                    if (onHandPosition == 1) & ((dfuse.iloc[i]['high'] - entryPrice) >= maxTP): #maxTP
                        onHandPosition = 0
                        df_position = pd.concat([df_position,pd.DataFrame({'position':[1],'profit':[maxTP],'model':[model],'note':['tp'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                        onHandPosition = 0
                        caseProtect = ''
                        priceProtect = 0
                        possibleLoss = 0
                        continue
                    
                    if (onHandPosition == -1) & ((entryPrice - dfuse.iloc[i]['low']) >= maxTP): #maxTP
                        onHandPosition = 0
                        df_position = pd.concat([df_position,pd.DataFrame({'position':[-1],'profit':[maxTP],'model':[model],'note':['tp'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                        onHandPosition = 0
                        caseProtect = ''
                        priceProtect = 0
                        possibleLoss = 0
                        continue
                           
                ## exit sl
                if df_condition.iloc[idxCondition]['exit sl'] == True:
                    if (onHandPosition == 1) & ((dfuse.iloc[i]['low'] - entryPrice) <= maxLoss): #maxTP
                        onHandPosition = 0
                        df_position = pd.concat([df_position,pd.DataFrame({'position':[1],'profit':[(dfuse.iloc[i]['low'] - entryPrice)],'model':[model],'note':['sl'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                        onHandPosition = 0
                        caseProtect = ''
                        priceProtect = 0
                        possibleLoss = 0
                        continue
                     
                    if (onHandPosition == -1) & ((entryPrice - dfuse.iloc[i]['high']) <= maxLoss): #maxTP
                        onHandPosition = 0
                        df_position = pd.concat([df_position,pd.DataFrame({'position':[-1],'profit':[(entryPrice - dfuse.iloc[i]['high'])],'model':[model],'note':['sl'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                        onHandPosition = 0
                        caseProtect = ''
                        priceProtect = 0
                        possibleLoss = 0
                        continue
                          
                ## exit limit
                if df_condition.iloc[idxCondition]['exit limit'] == True:
                    if (status['position'] == 0) & (status['count'] == maxHold):
                        if (onHandPosition == 1):
                            onHandPosition = 0
                            df_position = pd.concat([df_position,pd.DataFrame({'position':[1],'profit':[(dfuse.iloc[i]['low'] - entryPrice)],'model':[model],'note':['exit limit'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                            onHandPosition = 0
                            caseProtect = ''
                            priceProtect = 0
                            possibleLoss = 0
                            continue
            
                        if (onHandPosition == -1):
                            onHandPosition = 0
                            df_position = pd.concat([df_position,pd.DataFrame({'position':[-1],'profit':[(entryPrice - dfuse.iloc[i]['high'])],'model':[model],'note':['exit limit'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                            onHandPosition = 0
                            caseProtect = ''
                            priceProtect = 0
                            possibleLoss = 0
                            continue
        
                        
                ## exit reverse
                if df_condition.iloc[idxCondition]['exit reverse'] == True:
    
                    if (status['position'] == -1) & (onHandPosition == 1):
                        onHandPosition = 0
                        df_position = pd.concat([df_position,pd.DataFrame({'position':[1],'profit':[(dfuse.iloc[i]['low'] - entryPrice)],'model':[model],'note':['exit reverse'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                        onHandPosition = 0
                        caseProtect = ''
                        priceProtect = 0
                        possibleLoss = 0
                        continue
                        
                    if (status['position'] == 1) & (onHandPosition == -1):
                        onHandPosition = 0
                        df_position = pd.concat([df_position,pd.DataFrame({'position':[-1],'profit':[(entryPrice - dfuse.iloc[i]['high'])],'model':[model],'note':['exit reverse'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                        onHandPosition = 0
                        caseProtect = ''
                        priceProtect = 0
                        possibleLoss = 0
                        continue
                        
                
                ## exit tellingStopLoss tellingStopLoss protectBalance
                if (df_condition.iloc[idxCondition]['tellingStopLoss'] == True) | (df_condition.iloc[idxCondition]['protectBalance'] == True):
                    if caseProtect != '':
                        # print ('condition price')
    
                        if (onHandPosition == -1) & (dfuse.iloc[i]['high'] > priceProtect):
                            
                            df_position = pd.concat([df_position,pd.DataFrame({'position':[-1],'profit':[(entryPrice - priceProtect)],'model':[model],'note':['exit '+caseProtect],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                            onHandPosition = 0
                            caseProtect = ''
                            priceProtect = 0
                            possibleLoss = 0
                            continue
                            
                            
                        if (onHandPosition == 1) & (dfuse.iloc[i]['low'] < priceProtect):
                           
                           df_position = pd.concat([df_position,pd.DataFrame({'position':[1],'profit':[(priceProtect - entryPrice)],'model':[model],'note':['exit '+caseProtect],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                           onHandPosition = 0
                           caseProtect = ''
                           priceProtect = 0
                           possibleLoss = 0
                           continue
                
                ## update tellingStopLoss
                if (df_condition.iloc[idxCondition]['tellingStopLoss'] == True) & (caseProtect == 'protectBalance'):
                    # print ('tellingStopLoss')
                    if (onHandPosition == -1):
                        tmpTelling = ((entryPrice - dfuse.iloc[i]['close']) // df_condition.iloc[idxCondition]['priceProtect']) 
                        if tmpTelling >= 2:
                            tmpPrice = ((tmpTelling - 1) * df_condition.iloc[idxCondition]['priceProtect'] ) + (0.5 * df_condition.iloc[idxCondition]['priceProtect'])
                            tmpPrice = entryPrice - tmpPrice 
                            if tmpPrice < priceProtect:
                                priceProtect = tmpPrice
                                caseProtect = 'tellingStopLoss'
                    
                    if (onHandPosition == 1):
                        tmpTelling = (( dfuse.iloc[i]['close'] - entryPrice) // df_condition.iloc[idxCondition]['priceProtect']) 
                        if tmpTelling >= 2:
                            tmpPrice = ((tmpTelling - 1) * df_condition.iloc[idxCondition]['priceProtect'] ) + (0.5 * df_condition.iloc[idxCondition]['priceProtect'])
                            tmpPrice = entryPrice + tmpPrice 
                            if tmpPrice > priceProtect:
                                priceProtect = tmpPrice
                                caseProtect = 'tellingStopLoss'
                    
                
                ## update protectBalance
                if ((df_condition.iloc[idxCondition]['tellingStopLoss'] == True) | (df_condition.iloc[idxCondition]['protectBalance'] == True)) & (caseProtect == ''): 
                    if (onHandPosition == -1) & ((entryPrice - dfuse.iloc[i]['close']) >= df_condition.iloc[idxCondition]['priceProtect']):
                        caseProtect = 'protectBalance'
                        priceProtect = entryPrice-0.5
                        
                    if (onHandPosition == 1) & ((dfuse.iloc[i]['close'] - entryPrice) >= df_condition.iloc[idxCondition]['priceProtect']):
                        caseProtect = 'protectBalance'
                        priceProtect = entryPrice+0.5
                        
                
                ## entry
                if onHandPosition == 0:
                    if (status['position'] == 1) & (status['count'] >= entryPositionWhenCount):
                        onHandPosition = 1 
                        entryPrice = dfuse.iloc[i]['open']
                        entryTime = dfuse.iloc[i]['time_convert_x']
                        possibleLoss = 0
                    
                    elif (status['position'] == -1) & (status['count'] >= entryPositionWhenCount):
                        onHandPosition = -1 
                        entryPrice = dfuse.iloc[i]['open']
                        entryTime = dfuse.iloc[i]['time_convert_x']
                        possibleLoss = 0
                
                ## update possible loss
                if (status['position'] == 1):
                    tmpPsbLoss = dfuse.iloc[i]['low'] - entryPrice
                    if tmpPsbLoss < possibleLoss:
                        possibleLoss = tmpPsbLoss
                
                if (status['position'] == -1):
                    tmpPsbLoss = entryPrice - dfuse.iloc[i]['high']
                    if tmpPsbLoss < possibleLoss:
                        possibleLoss = tmpPsbLoss
                        
                ## force close last index
                if i == dfuse.shape[0] - 1:
                    if (onHandPosition == 1) & ((dfuse.iloc[i]['low'] - entryPrice) <= maxLoss): #maxTP
                        onHandPosition = 0
                        df_position = pd.concat([df_position,pd.DataFrame({'position':[1],'profit':[(dfuse.iloc[i]['low'] - entryPrice)],'model':[model],'note':['last index'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                        onHandPosition = 0
                        caseProtect = ''
                        priceProtect = 0
                        possibleLoss = 0
                        continue
                     
                    if (onHandPosition == -1) & ((entryPrice - dfuse.iloc[i]['high']) <= maxLoss): #maxTP
                        onHandPosition = 0
                        df_position = pd.concat([df_position,pd.DataFrame({'position':[-1],'profit':[(entryPrice - dfuse.iloc[i]['high'])],'model':[model],'note':['last index'],'entryTime':entryTime, 'exitTime':[dfuse.iloc[i]['time_convert_x']],'positionNote':[note],'possibleLoss':[possibleLoss]})])
                        onHandPosition = 0
                        caseProtect = ''
                        priceProtect = 0
                        possibleLoss = 0
                        continue
                    
                
                
    
    if df_position.shape[0] > 0 :
    
   
        df_position['holdPosition'] = pd.to_datetime(df_position['exitTime'])-pd.to_datetime(df_position['entryTime'])
        
        df_position.loc[df_position['profit']>0,'pnl'] = 'p'
        df_position.loc[df_position['profit']<=0,'pnl'] = 'l'
        
        
        df_position['key'] = df_position['positionNote']+ ' : ' + df_position['model']
        
        df_position = df_position.reset_index(drop=1)
        
        
        df_summary = pd.DataFrame()
        df_position2 = pd.DataFrame()
        
        for key in df_position['key'].unique():
            tmp = df_position.loc[df_position['key']==key].copy()
            
        
            # calmaxDD = tmp['profit']
            # cumulative_return = np.cumsum(calmaxDD)
        
            # # หา Maximum Drawdown
            # cumulative_peak = np.maximum.accumulate(cumulative_return)
            # drawdown = cumulative_return - cumulative_peak
            # max_drawdown = drawdown.min()
            
            tmp['cumulative_return'] = np.cumsum(tmp['profit'])
        
            # หา Maximum Drawdown
            tmp['cumulative_peak'] = np.maximum.accumulate(tmp['cumulative_return'])
            tmp['drawdown'] = tmp['cumulative_return'] - tmp['cumulative_peak']
            max_drawdown = tmp['drawdown'].min()
            
            
            tmp.loc[:,'est %dd'] = tmp['drawdown']/tmp['cumulative_return']
            
            df_position2 = pd.concat([df_position2,tmp])
            
            df_summary = pd.concat([df_summary, 
                                    pd.DataFrame({
                                        'key':[key],
                                        'profit':[tmp['profit'].sum()],
                                        'totalPosition':[tmp.shape[0]],
                                        'win':[tmp.loc[tmp['pnl']=='p'].shape[0]],
                                        'winRate':[tmp.loc[tmp['pnl']=='p'].shape[0] / tmp.shape[0]],
                                        'maxLoss':[tmp['profit'].min()],
                                        'maxDD':[max_drawdown],
                                        'est. %dd':[tmp['est %dd'].min()],
                                        'possibleLoss':[tmp['possibleLoss'].min()]
                                        })])
        
        df_summary = df_summary.reset_index(drop=1)         
        
        
        
        df_summary.to_csv( folderSummary +'/datasummary_'+str(idxCondition)+'.csv',index=0)
        df_position2.to_csv(folderPosition+'/dataposition_'+str(idxCondition)+'.csv',index=0)
    
    else:
        removePaths = [folderSummary+'/datasummary_'+str(idxCondition)+'.csv',
                       folderPosition+'/dataposition_'+str(idxCondition)+'.csv']
        for removePath in removePaths:
            
            if os.path.exists(removePath):
  
                os.remove(removePath)
        
    
    # print (idxCondition)
    return 1
    
    
# def worker(queue, progress_bar):
#     while not queue.empty():
#         # ดึงงานจาก queue
#         try:
#             i = queue.get_nowait()
#         except Exception:
#             break

#         # งานที่ต้องทำ
        
#         result = backtest(i)


#         # อัปเดต Progress Bar
#         progress_bar.update(1)

#         # แจ้งว่า queue เสร็จแล้ว
#         queue.task_done()
        
        

# indices = range (0 , df_condition.shape[0])





if __name__ == '__main__':
    # ไม่ต้องใส่ใน def
    
    
    start_time = time.time()
    
    
    
    os.makedirs(folderSummary, exist_ok=True)
    os.makedirs(folderPosition, exist_ok=True)
    
    
    useCPU = int(14) #cpu_count()
    
    print('use cpu:', useCPU)
    
    indices = range(0, df_condition.shape[0])
    # indices = range(263, df_condition.shape[0])
    # indices = range(0, 10)
    
    # args_list = [(idx, folderSummary, folderPosition) for idx in indices]
    
    # วิธีที่ 1: ใช้ Pool.map
    with Pool(processes= useCPU ) as pool: #cpu_count()
        with tqdm(total=len(indices), desc="Processing") as progress_bar:
            for result in pool.imap(backtest, indices ):
                progress_bar.update(1)
    
    # # หรือวิธีที่ 2: เก็บผลลัพธ์
    # with Pool(processes=cpu_count()) as pool:
    #     results = pool.map(backtest, indices)
    
    print("All done!")
    # print(f"Processed {len(results)} conditions")
    
    end_time = time.time()
    print(f"Multiprocessing completed in: {end_time - start_time:.2f} seconds")
    
    
    
    
    



# queue = Queue()

# for i in indices:
#     queue.put(i)


# print ('use cpu : ' , cpu_count())    
    
# with tqdm(total=len(indices), desc="Processing") as progress_bar:
#     # สร้าง thread pool
#     threads = []
#     for _ in range(cpu_count()):  
#         thread = threading.Thread(
#             target=worker, 
#             args=(queue,  progress_bar)
#         )
#         threads.append(thread)
#         thread.start()

#     # รอให้ thread ทุกตัวทำงานเสร็จ
#     for thread in threads:
#         thread.join()




















































