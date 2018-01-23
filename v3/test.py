
from dataRead import getData
from arch import ECG

import torch

if __name__ == "__main__":
    #### Path settings ####
    # Datapath
    data_path  = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/mitdb_data2s.npy'
    label_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/mitdb_label2s.npy'
    
    # Parameter settings
    BATCH_SIZE_TEST  = 64
    
    trainloader,testloader = getData(data_path,label_path,BATCH_SIZE_TRAIN,BATCH_SIZE_TEST)
    
    # Initialize Model
    ecg = ECG()
    ecg = ecg.cuda()