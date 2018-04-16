# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""
# Use the Azure Machine Learning data collector to log various metrics
from azureml.logging import get_azureml_logger
# Use the Azure Machine Learning data preparation package
from azureml.dataprep import package
from sklearn.cross_validation import train_test_split
import sys
import numpy as np
import pandas as pd
import gru4recModel as gru4rec
import evaluation

print("BEFORE MAIN")
if __name__ == '__main__':
    print("AFTER MAIN")
    logger = get_azureml_logger()

    # This call will load the referenced package and return a DataFrame.
    # If run in a PySpark environment, this call returns a
    # Spark DataFrame. If not, it will return a Pandas DataFrame.
    df = package.run('clickStreamPrep.dprep', dataflow_idx=0)
    print("Data loaded")
    data, valid = train_test_split(df, test_size=0.1)
    print("Data split")
    #Reproducing results from "Session-based Recommendations with Recurrent Neural Networks" on RSC15 (http://arxiv.org/abs/1511.06939)
    
    print('Training GRU4Rec with 100 hidden units')    
    
    gru = gru4rec.GRU4Rec(loss='top1', final_act='tanh', hidden_act='tanh', layers=[100], batch_size=50, dropout_p_hidden=0.5, learning_rate=0.01, momentum=0.0, time_sort=False)
    gru.fit(data)
    
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    
    
    #Reproducing results from "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations" on RSC15 (http://arxiv.org/abs/1706.03847)
    
    print('Training GRU4Rec with 100 hidden units')

    gru = gru4rec.GRU4Rec(loss='bpr-max-0.5', final_act='linear', hidden_act='tanh', layers=[100], batch_size=32, dropout_p_hidden=0.0, learning_rate=0.2, momentum=0.5, n_sample=2048, sample_alpha=0, time_sort=True)
    gru.fit(data)
    
    res = evaluation.evaluate_sessions_batch(gru, valid, None)
    print('Recall@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))