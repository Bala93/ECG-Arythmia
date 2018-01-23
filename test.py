
from torch.autograd import Variable
import torch

import numpy as np
from dataRead import getTestData
from tqdm import tqdm
#from arch import ECG

if __name__ == "__main__":
    #### Path settings ####
    # Datapath
    data_path  = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/mitdb_data2s.npy'
    label_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/mitdb_label2s.npy'
    model_path = '/media/htic/NewVolume1/murali/ecg/codes/memea/v1/models/Sat Jan 20 23:17:37 2018/Epoch24.pt'
    
    # Parameter settings
    BATCH_SIZE_TEST  = 64
    
    testloader = getTestData(data_path,label_path,BATCH_SIZE_TEST)
    
    # Initialize Model
    ecg = torch.load(model_path)
    ecg = ecg.cuda()
    
    
    
    accuracy_list = []
    
    for step,(x,y) in enumerate(tqdm(testloader)):
        x = Variable(x.cuda())
        y = y.cuda()
        y_predict = ecg(x)
        
        pred_y = torch.max(y_predict.data,1)[1]
        accu   = torch.sum(pred_y == y)/float(y.size()[0])
        accuracy_list.append(accu)    
    
    accuracy = np.mean(accuracy_list)
    print accuracy