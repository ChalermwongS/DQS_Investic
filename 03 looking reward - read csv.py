# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 23:26:21 2025

@author: RAI-CAD19
"""

from multiprocessing import  cpu_count
import threading
from queue import Queue
from tqdm import tqdm
import pandas as pd

from fn_charmy import *
from fn_Stategy import *

import time
import os

def worker(queue, dfRaw, df_dataLoad, progress_bar, results):
    while not queue.empty():
        # ดึงงานจาก queue
        try:
            i = queue.get_nowait()
        except Exception:
            break

        # งานที่ต้องทำ
        
        result = getReward(dfRaw, df_dataLoad, i)

        # print (i)

        # เก็บผลลัพธ์
        results.append(result.to_dict(orient='index')[0])

        # อัปเดต Progress Bar
        progress_bar.update(1)

        # แจ้งว่า queue เสร็จแล้ว
        queue.task_done()
        time.sleep(0.1)
        

# i = dfRaw[df_dataLoad['tf_text'][0]].loc[dfRaw[df_dataLoad['tf_text'][0]]['time']==1742573340].index[0]



def getReward(dfRaw, df_dataLoad, i):

        # dfCal = {}
        # dfCal[df_dataLoad['tf_text'][0]] = dfRaw[df_dataLoad['tf_text'][0]].iloc[i:i + df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0]].reset_index(drop=1)
        
        
        listLook = [3, 5, 10, 15, 30, 60, 120, 240, 480, 960, 1200, 1500]
        expectPF = 5
        
        reward = {}
        
        for idx, i_listLook in enumerate( listLook ):
            tmp = dfRaw[df_dataLoad['tf_text'][0]].iloc[i:i+i_listLook]
            # break
            base = tmp['close'].iloc[0]
            
            maxpfBuy = tmp['close'].iloc[1:].max()
            maxpfSell = tmp['close'].iloc[1:].min()
            
            tmpDDBuy = tmp.loc[(tmp['time']>tmp['time'].iloc[0]) & (tmp['time'] <= tmp.loc[(tmp['time']>tmp['time'].iloc[0]) & (tmp['close']== maxpfBuy) ]['time'].iloc[0])]['low'].min()
            tmpDDSell = tmp.loc[(tmp['time']>tmp['time'].iloc[0]) & (tmp['time'] <= tmp.loc[(tmp['time']>tmp['time'].iloc[0]) & (tmp['close']== maxpfSell)]['time'].iloc[0])]['high'].max()
            
            if (idx == 0) | (tmp.loc[tmp['close']== maxpfBuy].index[0] > listLook[idx-1]):
                reward[str(i_listLook)+':buy'] = maxpfBuy - base
                reward[str(i_listLook)+':buy:dd'] = tmpDDBuy - base
                if maxpfBuy - base >= expectPF:
                    reward[str(i_listLook)+':buy:rrr'] =  (maxpfBuy - base) / (base-tmpDDBuy) if (base-tmpDDBuy) != 0 else (maxpfBuy - base)
                else:
                    reward[str(i_listLook)+':buy:rrr'] = 0
            else:
                reward[str(i_listLook)+':buy'] = 0
                reward[str(i_listLook)+':buy:dd'] = 0
                reward[str(i_listLook)+':buy:rrr'] =  0
            
            if (idx == 0) | (tmp.loc[tmp['close']== maxpfSell].index[0] > listLook[idx-1]):
                reward[str(i_listLook)+':sell'] = base - maxpfSell
                reward[str(i_listLook)+':sell:dd'] = base - tmpDDSell
                if base - maxpfSell >= expectPF:
                    reward[str(i_listLook)+':sell:rrr'] =  (base - maxpfSell) / (tmpDDSell-base) if (tmpDDSell-base) != 0 else (base - maxpfSell)
                else:
                    reward[str(i_listLook)+':sell:rrr'] = 0
            else:
                reward[str(i_listLook)+':sell'] = 0
                reward[str(i_listLook)+':sell:dd'] = 0
                reward[str(i_listLook)+':sell:rrr'] =  0
        
        for idx, i_listLook in enumerate( listLook ):
            if (reward[str(i_listLook)+':buy:rrr'] > reward[str(i_listLook)+':sell:rrr']) &\
               (reward[str(i_listLook)+':buy:rrr'] >= 4) :
                   reward[str(i_listLook)+':act'] = 'buy'
            elif (reward[str(i_listLook)+':sell:rrr'] > reward[str(i_listLook)+':buy:rrr']) &\
               (reward[str(i_listLook)+':sell:rrr'] >= 4):
                   reward[str(i_listLook)+':act'] = 'sell'
            else:
                reward[str(i_listLook)+':act'] = '-'
        
        df_reward = pd.DataFrame([reward])
        
        df_time = tmp[['time','time_convert']].reset_index(drop=1).loc[:0]
      
        df_reward = pd.concat([df_time, df_reward],axis=1)
            
        # RRR = (ราคาที่ TP – ราคาจุดที่เข้า) / (ราคาจุดที่เข้า – ราคาที่ SL)
        
        return df_reward
            
        
        
        
        
        

def main(tmp, split):
    # login()

    # dfRaw = {}
    # for idx, tf_text in enumerate( df_dataLoad['tf_text']):
    #     dfRaw[tf_text] = {}
    #     dfRaw[tf_text] = loadData(tf = df_dataLoad['tf'][idx], callBackBar=99999)
    
    dfRaw = {}
    folder_path = 'data csv'
    for filename in os.listdir(folder_path):
        dfRaw[filename.replace('.csv','')] = pd.read_csv(folder_path+'/'+filename)
       

    # indices = range ( df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0]  , 
    #                  dfRaw[df_dataLoad['tf_text'][0]].shape[0] - df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0] )    
    
    # indices = range (0,50)
    
    if tmp*split+split < dfRaw[df_dataLoad['tf_text'][0]].shape[0] - df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0]:
        indices = range (tmp*split, tmp*split+split)
    else:
        indices = range (tmp*split, dfRaw[df_dataLoad['tf_text'][0]].shape[0] - df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0])
    
    print (tmp , indices)
    
    
    queue = Queue()
    results = []

    # เติม queue ด้วย index
    for i in indices:
        queue.put(i)
    
    with tqdm(total=len(indices), desc="Processing") as progress_bar:
        # สร้าง thread pool
        threads = []
        for _ in range(cpu_count()):  # ใช้ cpu_count threads
            thread = threading.Thread(
                target=worker, 
                args=(queue, dfRaw, df_dataLoad, progress_bar, results)
            )
            threads.append(thread)
            thread.start()

        # รอให้ thread ทุกตัวทำงานเสร็จ
        for thread in threads:
            thread.join()

    # รวมผลลัพธ์
    dfResults = pd.DataFrame(results)
    return dfResults.sort_values('time').reset_index(drop=1) 

if __name__ == "__main__":
    
    startTime = time.time()
    
    tmp = 3
    split = 1000000
    
    result = main(tmp, split)     
    
    # result = main()        
    
    print ('processTime',time.time()-startTime)
    
    # result.to_csv('rewardResult.csv')
    
    # result.to_csv('datareward/rewardResult_'+str(tmp)+'.csv')    

    result.to_csv('datareward/rewardResult_'+str(tmp)+'.csv.gz',compression = 'gzip')
    
    
    
    
    
    
    
    
    
    
    
    