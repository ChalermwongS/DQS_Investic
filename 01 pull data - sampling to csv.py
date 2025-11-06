# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 10:21:25 2025

@author: Chalermwong
"""

import pandas as pd

from fn_charmy import *
from fn_Stategy import *



login()

dfRaw = {}
for idx, tf_text in enumerate( df_dataLoad['tf_text']):
    dfRaw[tf_text] = {}
    dfRaw[tf_text] = loadData(tf = df_dataLoad['tf'][idx], callBackBar=3800000)
    
    # dfRaw[tf_text].to_csv('data csv/'+tf_text+'.csv',index=0)
    
    tmp = pd.read_csv('data csv/'+tf_text+'.csv')
    
    tmp1 = dfRaw[tf_text].loc[dfRaw[tf_text]['time'] >= tmp['time'].max()]
    tmp2 = pd.concat([tmp,tmp1])
    tmp2 = tmp2.drop_duplicates(subset='time',keep='last')
    tmp2.to_csv('data csv/'+tf_text+'.csv',index=0)
    
    
    
    
# a = pd.DataFrame()
# a['time_convert_th'] = dfRaw[df_dataLoad['tf_text'][0]]['time_convert_th']
# a['date'] = dfRaw[df_dataLoad['tf_text'][0]]['time_convert_th'].dt.date
# a = a.drop_duplicates(subset='date')