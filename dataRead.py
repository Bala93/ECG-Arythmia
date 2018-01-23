#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:20:38 2018

@author: htic
"""
import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset



def getData(data_path,label_path,BATCH_SIZE_TRAIN,BATCH_SIZE_TEST):
    
    in_data_npy = np.load(data_path)
    out_data_npy = np.load(label_path)
    
    
    # Randomly pick 2500 dataset as it is the lowest among all. 
    
    label_0 = np.where(out_data_npy == 0)
    label_1 = np.where(out_data_npy == 1)
    label_2 = np.where(out_data_npy == 2)
    
    in_data_0 = in_data_npy[label_0]
    in_data_1 = in_data_npy[label_1]
    in_data_2 = in_data_npy[label_2]
   
    
    # Random shuffle is done and first 2500 elements are taken. 
    # First 2000 will be taken for train data and the remaining will be taken for test data. 
    
    CLASS_COUNT = 2500
    INDEX_SEPARATE = 2000
    TRAIN_LEN = 2000
    TEST_LEN  = 500
    
    np.random.shuffle(in_data_0)
    np.random.shuffle(in_data_1)
    np.random.shuffle(in_data_2)
    
    in_data_0 = in_data_0[:CLASS_COUNT,:]
    in_data_1 = in_data_1[:CLASS_COUNT,:]
    in_data_2 = in_data_2[:CLASS_COUNT,:]
    
    train_data_0 = in_data_0[:INDEX_SEPARATE,:]
    train_data_1 = in_data_1[:INDEX_SEPARATE,:]
    train_data_2 = in_data_2[:INDEX_SEPARATE,:]
    
    train_label_0 = np.zeros([TRAIN_LEN,1])
    train_label_1 = np.ones([TRAIN_LEN,1])
    train_label_2 = 2 * np.ones([TRAIN_LEN,1])

    
    test_data_0 = in_data_0[INDEX_SEPARATE:,:]
    test_data_1 = in_data_1[INDEX_SEPARATE:,:]
    test_data_2 = in_data_2[INDEX_SEPARATE:,:]

    test_label_0 = np.zeros([TEST_LEN,1])
    test_label_1 = np.ones([TEST_LEN,1])
    test_label_2 = 2 * np.ones([TEST_LEN,1])

    
    train_data  = np.vstack((train_data_0,train_data_1,train_data_2))
    test_data   = np.vstack((test_data_0,test_data_1,test_data_2))
    train_label = np.vstack((train_label_0,train_label_1,train_label_2))
    test_label  = np.vstack((test_label_0,test_label_1,test_label_2))
    
    # Shuffle the dataset 
    
    train_data_label = np.hstack((train_data,train_label))
    test_data_label  = np.hstack((test_data,test_label))
    
    np.random.shuffle(train_data_label)
    np.random.shuffle(test_data_label)
    
    trainData  = train_data_label[:,:-1]
    trainLabel = train_data_label[:,-1]
    testData   = test_data_label[:,:-1]
    testLabel  = test_data_label[:,-1]

    print "Dataset Distribution", np.unique(out_data_npy,return_counts=1)
    print "Random 2500 from each class is taken and 2000 is used for train and remaining for test" 
    print "No of Training data", train_label.shape[0]
    print "No of Testing data",test_label.shape[0]
    print "Feature length", train_data.shape[1]    

    
    # Converting to torch tensors
    
    train_data_torch = torch.FloatTensor(trainData)
    train_data_torch = train_data_torch.unsqueeze(1)
    train_label_torch = torch.LongTensor(trainLabel)
    
    test_data_torch = torch.FloatTensor(testData)
    test_data_torch = test_data_torch.unsqueeze(1)
    test_label_torch = torch.LongTensor(testLabel)


    # Creating a dataloader
    
    train_dataset = TensorDataset(train_data_torch,train_label_torch)
    trainLoader  = DataLoader(train_dataset,batch_size=BATCH_SIZE_TRAIN)
    
    test_dataset  = TensorDataset(test_data_torch,test_label_torch)
    testLoader   = DataLoader(test_dataset,batch_size=BATCH_SIZE_TEST)
    
    
    return trainLoader,testLoader


def getTestData(data_path,label_path,BATCH_SIZE_TEST):
    
    in_data_npy = np.load(data_path)
    out_data_npy = np.load(label_path)
    
    # Randomly pick 500 dataset as it is the lowest among all. 
    
    label_0 = np.where(out_data_npy == 0)
    label_1 = np.where(out_data_npy == 1)
    label_2 = np.where(out_data_npy == 2)
    
    in_data_0 = in_data_npy[label_0]
    in_data_1 = in_data_npy[label_1]
    in_data_2 = in_data_npy[label_2]
   
    TEST_LEN  = 500
    
    np.random.shuffle(in_data_0)
    np.random.shuffle(in_data_1)
    np.random.shuffle(in_data_2)
    
    
    test_data_0 = in_data_0[:TEST_LEN,:]
    test_data_1 = in_data_1[:TEST_LEN,:]
    test_data_2 = in_data_2[:TEST_LEN,:]
    test_data   = np.vstack((test_data_0,test_data_1,test_data_2))
    
    test_label_0 = np.zeros([TEST_LEN,1])
    test_label_1 = np.ones([TEST_LEN,1])
    test_label_2 = 2 * np.ones([TEST_LEN,1])
    test_label  = np.vstack((test_label_0,test_label_1,test_label_2))
    
    test_data_label  = np.hstack((test_data,test_label))
    np.random.shuffle(test_data_label)
    
    testData   = test_data_label[:,:-1]
    testLabel  = test_data_label[:,-1]
    
    test_data_torch = torch.FloatTensor(testData)
    test_data_torch = test_data_torch.unsqueeze(1)
    test_label_torch = torch.LongTensor(testLabel)
    
    test_dataset  = TensorDataset(test_data_torch,test_label_torch)
    testLoader   = DataLoader(test_dataset,batch_size=BATCH_SIZE_TEST)
    
    return testLoader

    
#data_path  = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/mitdb_data2s.npy'
#label_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/mitdb_label2s.npy'
#
#getData(data_path,label_path)