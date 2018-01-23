#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:25:13 2018

@author: htic
"""

from torch import nn


INPUT_SIZE       = 72
TIME_STEP        = 10
HIDDEN_UNIT_SIZE = 40
NO_CLASS = 3

class ECG(nn.Module):
    
    def __init__(self):
        super(ECG,self).__init__()
        
        self.lstm = nn.LSTM(
                input_size  = INPUT_SIZE,
                hidden_size = HIDDEN_UNIT_SIZE,
                num_layers  = 2,
                batch_first = True,
#                bidirectional = True
                )
        
        self.out = nn.Linear(HIDDEN_UNIT_SIZE,NO_CLASS)
        
    
    def forward(self,x):
        
        rout,(h_n,h_c) = self.lstm(x,None)   
        rout = rout[:,-1,:].unsqueeze(1)
#        print rout.size()
        out = self.out(rout)
        out = out.squeeze(1)
        return out