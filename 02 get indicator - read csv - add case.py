# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 23:54:38 2025

@author: Chalermwong
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
        
        result = getStats(dfRaw, df_dataLoad, i)

        # print (i)

        # เก็บผลลัพธ์
        results.append(result.to_dict(orient='index')[0])

        # อัปเดต Progress Bar
        progress_bar.update(1)

        # แจ้งว่า queue เสร็จแล้ว
        queue.task_done()


def getStats(dfRaw, df_dataLoad, i):
        
        i = i - dfRaw[df_dataLoad['tf_text'][0]].index.min() #for shift index to start point from selct use only
        dfCal = {}
        dfCal[df_dataLoad['tf_text'][0]] = dfRaw[df_dataLoad['tf_text'][0]].iloc[i:i + df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0]].reset_index(drop=1)
        
        for idx_tf in range (1, df_dataLoad.tf.shape[0]):
            tmplook = 1 if idx_tf != 2 else 2 # idx_tf ==2 is 5 min --> must be compare with 1 min not 3 min
            
            dfCal[df_dataLoad['tf_text'][idx_tf]] = dfRaw[df_dataLoad['tf_text'][idx_tf]].loc[dfRaw[df_dataLoad['tf_text'][idx_tf]]['time'] <= dfCal[df_dataLoad['tf_text'][idx_tf-tmplook]]['time'].max()]\
                .iloc[- df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][idx_tf]]['select'].reset_index(drop=1)[0] :].reset_index(drop=1)
            
            tmpSmallTf = dfCal[df_dataLoad['tf_text'][idx_tf-tmplook]].loc[dfCal[df_dataLoad['tf_text'][idx_tf-tmplook]]['time']>=dfCal[df_dataLoad['tf_text'][idx_tf]]['time'].max()]
            
            dfCal[df_dataLoad['tf_text'][idx_tf]].loc[dfCal[df_dataLoad['tf_text'][idx_tf]]['time'] == dfCal[df_dataLoad['tf_text'][idx_tf]]['time'].max(),'open'] = tmpSmallTf.loc[tmpSmallTf['time'] == tmpSmallTf['time'].min()]['open'].reset_index(drop=1)[0]
            dfCal[df_dataLoad['tf_text'][idx_tf]].loc[dfCal[df_dataLoad['tf_text'][idx_tf]]['time'] == dfCal[df_dataLoad['tf_text'][idx_tf]]['time'].max(),'high'] = tmpSmallTf.loc[tmpSmallTf['high'] == tmpSmallTf['high'].max()]['high'].reset_index(drop=1)[0]
            dfCal[df_dataLoad['tf_text'][idx_tf]].loc[dfCal[df_dataLoad['tf_text'][idx_tf]]['time'] == dfCal[df_dataLoad['tf_text'][idx_tf]]['time'].max(),'low'] = tmpSmallTf.loc[tmpSmallTf['low'] == tmpSmallTf['low'].min()]['low'].reset_index(drop=1)[0]
            dfCal[df_dataLoad['tf_text'][idx_tf]].loc[dfCal[df_dataLoad['tf_text'][idx_tf]]['time'] == dfCal[df_dataLoad['tf_text'][idx_tf]]['time'].max(),'close'] = tmpSmallTf.loc[tmpSmallTf['time'] == tmpSmallTf['time'].max()]['close'].reset_index(drop=1)[0] 
        
        
        ### process ###
        
        dfStats = pd.DataFrame()
        dfStats['tf_text'] = dfCal.keys()
        dfStats['trend'] = ''
        dfStats['trendCross'] = ''
        dfStats['ratioValue'] = 0.0
        
        dfStats['bullish'] = ''
        dfStats['bearish'] = ''
        dfStats['hiddenBullish'] = ''
        dfStats['hiddenBearish'] = ''
        dfStats['testSupport'] = ''
        dfStats['testStorngSupport'] = ''
        dfStats['testResistance'] = ''
        dfStats['testStorngResistance'] = ''
        
        dfStats['SAR-Up'] = ''
        dfStats['SAR-Down'] = ''
        dfStats['Sto'] = ''
        dfStats['StoReversedLow'] = ''
        dfStats['StoReversedHigh'] = ''
        
        
        for tf_text in dfCal.keys():
            dfCal[tf_text] = pivotPointAndStorngZone(dfCal[tf_text],
                                                df_dataCondition.loc[df_dataCondition['tf_text']==tf_text].reset_index(drop=1)['order'][0],
                                                df_dataCondition.loc[df_dataCondition['tf_text']==tf_text].reset_index(drop=1)['orderSmall'][0] )
            dfCal[tf_text] = find_divergence(dfCal[tf_text])
            dfCal[tf_text] = normalIndicator(dfCal[tf_text])
            
            dfCal[tf_text], trend, trendCross ,ratioValue = calTrend(dfCal[tf_text])
            
            dfStats.loc[dfStats['tf_text']==tf_text,'trend'] = trend
            dfStats.loc[dfStats['tf_text']==tf_text,'trendCross'] = trendCross
            dfStats.loc[dfStats['tf_text']==tf_text,'ratioValue'] = ratioValue
            
            lookBack = df_dataCondition.loc[df_dataCondition['tf_text']==tf_text]['lookBackIndicator'].iloc[0]
            
            
            for keyIndicator in ['bullish', 'bearish', 'hiddenBullish', 'hiddenBearish']:
                if dfCal[tf_text].iloc[-lookBack: ][keyIndicator].sum() > 0:
                    dfStats.loc[dfStats['tf_text']==tf_text,keyIndicator] = True
            
            testZone = checkTestZone(dfCal[tf_text],lookBack)
            for idx, key in enumerate( ['testSupport', 'testStorngSupport', 'testResistance', 'testStorngResistance'] ):
                dfStats.loc[dfStats['tf_text']==tf_text, key] = testZone[idx]
                
            
            if dfCal[tf_text]['PSARl_0.02_0.2'].iloc[-1:].isna().sum()==1:
                dfStats.loc[dfStats['tf_text']==tf_text,'SAR-Up'] = True
                
            if dfCal[tf_text]['PSARs_0.02_0.2'].iloc[-1:].isna().sum()==1:
                dfStats.loc[dfStats['tf_text']==tf_text,'SAR-Down'] = True
                
            Sto, StoReversedLow, StoReversedHigh = calStoReversed(dfCal[tf_text])
            dfStats.loc[dfStats['tf_text']==tf_text,'Sto'] = Sto
            dfStats.loc[dfStats['tf_text']==tf_text,'StoReversedLow'] = StoReversedLow
            dfStats.loc[dfStats['tf_text']==tf_text,'StoReversedHigh'] = StoReversedHigh
            
            
        dfStats2 = pd.melt(dfStats, 
                  id_vars=['tf_text'], 
                  var_name='tf_field', 
                  value_name='value')
        
        dfStats2['key'] = dfStats2['tf_text'] + ':' + dfStats2['tf_field']
        dfStats2 = dfStats2[['key','value']]
        dfStats2 = dfStats2.T
        dfStats2.columns = dfStats2.iloc[0]
        dfStats2 = dfStats2.reset_index(drop=True).drop(index=0)
        
        dfStats2 = pd.concat([dfCal[df_dataLoad['tf_text'][0]][['time','time_convert']].iloc[-1:].reset_index(drop=True),dfStats2.reset_index(drop=True)], axis=1)
    
        
        return dfStats2
    
    
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
   
    
    
def main(tmp, split, lastTime):
    
    # login()

    # dfRaw = {}
    # for idx, tf_text in enumerate( df_dataLoad['tf_text']):
    #     dfRaw[tf_text] = {}
    #     dfRaw[tf_text] = loadData(tf = df_dataLoad['tf'][idx], callBackBar=3800000)
    dfRaw = {}
    folder_path = 'data csv'
    for filename in os.listdir(folder_path):
        df_tmp = pd.read_csv(folder_path+'/'+filename)
        df_tmp, report_df = optimize_df(df_tmp)
        dfRaw[filename.replace('.csv','')] = df_tmp
        # dfRaw[filename.replace('.csv','')] = pd.read_csv(folder_path+'/'+filename)
        
    dfRaw[df_dataLoad['tf_text'][0]] = dfRaw[df_dataLoad['tf_text'][0]].loc[dfRaw[df_dataLoad['tf_text'][0]]['time']>lastTime].reset_index(drop=1)
    
    # indices = range (dfRaw[df_dataLoad['tf_text'][0]].shape[0] - df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0] )    
    
    if tmp*split+split < dfRaw[df_dataLoad['tf_text'][0]].shape[0] - df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0]:
        indices = range (tmp*split, tmp*split+split)
    else:
        indices = range (tmp*split, dfRaw[df_dataLoad['tf_text'][0]].shape[0] - df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0])
    
    #select use only
    dfRaw[df_dataLoad['tf_text'][0]] = dfRaw[df_dataLoad['tf_text'][0]].iloc[ indices[0] :  (indices[-1]+1 + df_dataLoad.loc[df_dataLoad['tf_text']==df_dataLoad['tf_text'][0]]['select'].reset_index(drop=1)[0] )]
    
    
    print (tmp , indices)
    # indices = range (0,50)
    # indices = range(400000-30, 400000)
    
    
    queue = Queue()
    results = []

    # เติม queue ด้วย index
    for i in indices:
        queue.put(i)
    
    with tqdm(total=len(indices), desc="Processing") as progress_bar:
        # สร้าง thread pool
        threads = []
        for _ in range(cpu_count()):  # ใช้ 8 threads
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
    
    df_indicator = pd.DataFrame()
    folder_path = 'dataindicator'
    for filename in os.listdir(folder_path):
        df_indicator = pd.concat([df_indicator,pd.read_csv(folder_path+'/'+filename,low_memory=False,usecols=['time'])])

    df_indicator['time'].max()
    
    tmp = 0
    split = 15000
    
    result = main(tmp, split, df_indicator['time'].max())        
    
    print ('processTime',time.time()-startTime)
        
    result.to_csv('dataindicator/indicatorResult_1_'+str(tmp)+'.csv')    
        
        
    

# results = []

# a = getStats(dfRaw, df_dataLoad, i)

# results.append(a.to_dict(orient='index')[0])
# results.append(a.to_dict(orient='index')[0])

# dfResults = pd.DataFrame(results)
# dfResults
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    