# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
# Use the Azure Machine Learning data collector to log various metrics
from azureml.logging import get_azureml_logger
# Use the Azure Machine Learning data preparation package
from azureml.dataprep import package
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

import gru4rec_tensor as model
import eval_tensor as evaluation

class Args():
    is_training = True
    layers = 1
    rnn_size = 100
    n_epochs = 3
    batch_size = 50
    dropout_p_hidden=1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './outputs'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1

def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    
    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()
    logger = get_azureml_logger()
    # This call will load the referenced package and return a DataFrame.
    # If run in a PySpark environment, this call returns a
    # Spark DataFrame. If not, it will return a Pandas DataFrame.
    df = package.run('clickStreamPrep.dprep', dataflow_idx=0)
    print("Data loaded")
    #data = df
    #valid = df
    data, valid = train_test_split(df, test_size=0.3, shuffle=False)
    args = Args()
    #Where is n_items set/used etc?
    itemIds = df['ItemId'].unique()
    args.n_items = len(itemIds)
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    #args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    print(args.dropout_p_hidden)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        #oSaver = tf.train.Saver()
        gru = model.GRU4Rec(sess, args)
        if args.is_training:
            gru.fit(data, args.n_items, itemIds)
            #oSaver.save(sess, './outputs/model.cktp')
        else:
            #oSaver.restore(sess, './outputs/model.cktp')
            res = evaluation.evaluate_sessions_batch(gru, data, valid, itemIds)
            print('Recall@20: {}\tMRR@20: {}'.format(res[0], res[1]))
        
