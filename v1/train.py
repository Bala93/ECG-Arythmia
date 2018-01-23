#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:03:12 2018

@author: htic
"""

## Common Modules
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
import time
import os
import copy

# Written functions
from arch import ECG
from dataRead import getData



def testDataEval(testloader,ecg):
    accuracy_list = []
    
    for step,(x,y) in enumerate(testloader):
        x = Variable(x.cuda())
        y = y.cuda()
        y_predict = ecg(x)
        
        pred_y = torch.max(y_predict.data,1)[1]
        accu   = torch.sum(pred_y == y)/float(y.size()[0])
        accuracy_list.append(accu)    
    
    accuracy = np.mean(accuracy_list)

    return accuracy


if __name__ == "__main__":

    
    #### Path settings ####
    # Datapath
    data_path  = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/mitdb_data2s.npy'
    label_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/mitdb_label2s.npy'
    
    #Logfiles
    start_time = time.ctime()
    log_file = './logfiles/' + start_time + '.txt'
    log_data = open(log_file,'w')
    model_path = './models/' + start_time + '/'
    os.mkdir(model_path)
    
    # Parameter settings
    EPOCH = 25
    LR    = 0.01
    BATCH_SIZE_TRAIN = 8
    BATCH_SIZE_TEST  = 64
    
    # Dataloaders
    trainloader,testloader = getData(data_path,label_path,BATCH_SIZE_TRAIN,BATCH_SIZE_TEST)
    
    # Initialize Model
    ecg = ECG()
    ecg = ecg.cuda()

    #Loss function
    optimizer = torch.optim.SGD(ecg.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()
    
    # Training 
    for epoch in range(EPOCH):
        for step,(x,y) in enumerate(trainloader):
            
            x,y = Variable(x.cuda()),Variable(y.cuda())
            y_predict = ecg(x)
            loss = loss_func(y_predict,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if step % 100 == 0:
                acc = testDataEval(testloader,ecg)
                update = 'Epoch: ', epoch, '| step : %d' %step,  ' | train loss : %.6f' % loss.data[0], '| test accuracy: %.4f' % acc
                update_str = [str(i) for i in update]
                update_str = ''.join(update_str)	
                print update_str
                log_data.write(update_str + '\n')
                

        print "Saving the model" 
        torch.save(ecg,model_path+'Epoch'+str(epoch)+'.pt')
