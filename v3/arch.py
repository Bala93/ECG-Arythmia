#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:08:51 2018

@author: htic
"""
import torch.nn as nn
import torch


FLATTEN_DIM = 640
INPUT_SIZE  = 60
TIME_STEP   = 12
HIDDEN_UNIT_SIZE = 40
CAT_SIZE = FLATTEN_DIM + HIDDEN_UNIT_SIZE
NO_CLASS = 3


class ECG(nn.Module):
    
    def __init__(self):
        super(ECG,self).__init__()
        
        convsize    = 15
        convpadding  = 7
        poolsize = 2
        poolstr  = 2
        convstr  = 1
#        in_feature_count = 1
#        out_feature_count = 2
        in_size  = 64 
        out_size = 64
        flag = False
        
        #self.conv1 = nn.Conv1d(in_channels=1,out_channels=64,kernel_size=convsize,stride=convstr,padding=convpadding)

    	# Sharath layers like inception
        ##############
        self.conv_incep_1 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=convsize,stride=convstr,padding=convpadding)
        self.conv_incep_2 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=convsize+2,stride=convstr,padding=convpadding+1)
        self.conv_incep_3 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=convsize+4,stride=convstr,padding=convpadding+2)
        self.conv_incep_4 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=convsize+6,stride=convstr,padding=convpadding+3)
        ###############


        self.bn1   = nn.BatchNorm1d(64)
        self.rel1  = nn.ReLU()
       
        self.mp1   = nn.MaxPool1d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv1d(64,64,kernel_size=convsize,stride=1,padding=convpadding)
        self.bn2   = nn.BatchNorm1d(64)
        self.rel2  = nn.ReLU()
        self.dp1   = nn.Dropout()
        self.conv3 = nn.Conv1d(64,64,kernel_size=convsize,stride=1,padding=convpadding)
        self.mp2   = nn.MaxPool1d(kernel_size=poolsize,stride=poolstr)
        self.res2 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        
        
        flag = not flag
        self.res3 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        self.mp3   = nn.MaxPool1d(kernel_size=poolsize,stride=poolstr)
        
        
        in_size = 64
        out_size = 64 * 2
        flag = not flag
        self.conv_same_1 = nn.Conv1d(in_size,out_size,kernel_size=1,stride=1)
        self.res4 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
    
        
        in_size = 64 * 2
        flag = not flag
        self.res5 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        self.mp4  = nn.MaxPool1d(kernel_size=poolsize,stride=poolstr)
        
        flag = not flag
        self.res6 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        
        
        flag = not flag
        self.res7 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)        
        self.mp5  = nn.MaxPool1d(kernel_size=poolsize,stride=poolstr)
        
        flag = not flag
        in_size = 64 * 2
        out_size = 64 * 3
        self.conv_same_2 = nn.Conv1d(in_size,out_size,kernel_size=1,stride=1)
        self.res8 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        
        
        flag = not flag
        in_size = 64 * 3
        self.res9 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        self.mp6  = nn.MaxPool1d(kernel_size=poolsize,stride=poolstr)  
        
        flag = not flag
        self.res10 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)        
        
        flag = not flag
        self.res11 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        self.mp7  = nn.MaxPool1d(kernel_size=poolsize,stride=poolstr)
        
        in_size = 64 * 3
        out_size = 64 * 4
        flag = not flag
        self.conv_same_3 = nn.Conv1d(in_size,out_size,kernel_size=1,stride=1)
        self.res12 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        
        in_size = 64 * 4
        flag = not flag
        self.res13 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)        
        self.mp8  = nn.MaxPool1d(kernel_size=poolsize,stride=poolstr)
        
        flag = not flag
        self.res14 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        
        
        flag = not flag
        self.res15 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)
        self.mp9  = nn.MaxPool1d(kernel_size=poolsize,stride=poolstr)
        
        flag = not flag
        in_size = 64 * 4
        out_size = 64 * 5
        self.conv_same_4 = nn.Conv1d(in_size,out_size,kernel_size=1,stride=1)
        self.res16 = self.resblock(in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag)        
        self.bn_n = nn.BatchNorm1d(out_size)
        
        
        ### RNN ###
        self.lstm = nn.LSTM(
                    input_size  = INPUT_SIZE,
                    hidden_size = HIDDEN_UNIT_SIZE,
                    num_layers  = 2,
                    batch_first = True)

        
    	self.fc1 = nn.Linear(CAT_SIZE,500)
        self.fc2 = nn.Linear(500,100)
        self.fc3 = nn.Linear(100,3)
        

    def resblock(self,in_size,out_size,convsize,convpadding,convstr,poolsize,poolstr,flag):
        
        layers = []
        layers.append(nn.BatchNorm1d(in_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout())
        layers.append(nn.Conv1d(in_size,out_size,kernel_size=convsize,stride=convstr,padding=convpadding))
        layers.append(nn.BatchNorm1d(out_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout())
        layers.append(nn.Conv1d(out_size,out_size,kernel_size=convsize,stride=convstr,padding=convpadding))
        
        if flag:
            layers.append(nn.MaxPool1d(kernel_size=poolsize,stride=poolstr))
       
        block = nn.Sequential(*layers)
    
        return block
    
    def forward(self,x):
	 
        inp_x = x.view(-1,TIME_STEP,INPUT_SIZE)
        #x = self.conv1(x)
        #print x.size()
    	inc_x1 = self.conv_incep_1(x)
        #print inc_x1.size()
    	inc_x2 = self.conv_incep_2(x)
        #print inc_x2.size()
    	inc_x3 = self.conv_incep_3(x)
        #print inc_x3.size()
    	inc_x4 = self.conv_incep_4(x)
        #print inc_x4.size()
    	x = torch.cat((inc_x1,inc_x2,inc_x3,inc_x4),1)
        #print x.size()	
        #print "Conv1:",x.size()
        x = self.bn1(x)
        x = self.rel1(x)
        x1 = x
        x = self.conv2(x)
        #print "Left path conv2:",x.size()
        x = self.bn2(x)
        x = self.rel2(x)
        x = self.dp1(x)
        x = self.conv3(x)
        #print "Left path conv3:",x.size()
        x = self.mp2(x)
       	#print "Left path maxpool:",x.size() 
        x1 = self.mp1(x1)
        #print "Right path maxpool:",x1.size()
        x += x1
        
    
        x2 = x
        x = self.res2(x)
        #print "Left --> Res block 2:",x.size()
        #print "Right --> Res block 2:",x.size()
        x += x2
       
        x3 = self.mp3(x)
        x = self.res3(x)
        #print "Left --> Res block 3:",x.size()
        #print "Right --> Res block 3:",x3.size()
        x += x3
       
	
        x4 = self.conv_same_1(x) 
        x = self.res4(x)
        #print "Left --> Res block 4:",x.size()
        #print "Right --> Res block 4:",x4.size()
        x += x4
        
        
	
        x5 = self.mp4(x)
        x = self.res5(x)
        #print "Left --> Res block 5:",x.size()
        #print "Right --> Res block 5:",x5.size()
        x += x5
        
        x6 = x
        x = self.res6(x)
        #print "Left --> Res block 6:",x.size()
        #print "Right --- > Res block 6:",x6.size()
        x += x6
       
	 
        x7 = self.mp5(x)
        x = self.res7(x)
        #print "Left --> Res block 7:",x.size()
        #print "Right -- > Res block 7:",x7.size()
        x += x7
    

        x8 = self.conv_same_2(x)
        x = self.res8(x)
        #print "Left --> Res block 8:",x.size()
        #Print "Right -- > Res block 8:",x8.size()
        x += x8

        
        x9 = self.mp6(x)
        x = self.res9(x)
        #print "Left --> Res block 9:",x.size()
        #print "Right -- > Res block 9:",x9.size()
        x += x9
        
        x10 = x
        x = self.res10(x)
        #print "Left --> Res block 10:",x.size()
        #print "Right -- > Res block 10:",x10.size()
        x += x10

        x11 = self.mp7(x)
        x = self.res11(x)
        #print "Left --> Res block 11:",x.size()
        #print "Right -- > Res block 11:",x11.size()
        x += x11
        
        x12 = self.conv_same_3(x)
        x = self.res12(x)
        #print "Left --> Res block 12:",x.size()
        #print "Right -- > Res block 12:",x12.size()
        x += x12
       
        x13 = self.mp8(x)
        x = self.res13(x)
        #print "Left --> Res block 13:",x.size()
        #print "Right -- > Res block 13:",x13.size()
        x += x13
        
        x14 = x
        x = self.res14(x)
        #print "Left --> Res block 14:",x.size()
        #print "Right -- > Res block 14:",x14.size()
        x += x14
        
        x15 = self.mp9(x)
        x = self.res15(x)
        #print "Left --> Res block 15:",x.size()
        #print "Right -- > Res block 15:",x15.size()
        x += x15
	
    	x16 = self.conv_same_4(x)
        x = self.res16(x)
        x += x16
	
    	x = self.bn_n(x)
        #print x.size()
    	x = x.view(-1,FLATTEN_DIM)
        
        rout,(h_n,h_c) = self.lstm(inp_x,None)
        rout = rout[:,-1,:]
        
        x = torch.cat((x,rout),dim=1)
    	x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.fc4(x)
        return x

    
#model = ECG()
#print model
#model = model.cuda()
#x = torch.randn([1,1,720])
#x = x.cuda()
#x = Variable(x)
#y = model(x)
