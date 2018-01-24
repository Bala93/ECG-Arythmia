
from torch.autograd import Variable
import torch
import numpy as np
from dataRead import getTestData
from tqdm import tqdm
from matplotlib import pyplot as plt



if __name__ == "__main__":
    #### Path settings ####
    # Datapath
    
    data_path  = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/ltdb2_data2s.npy'
    label_path = '/media/htic/NewVolume1/murali/ecg/codes/datasets/multidataset/ltdb2_label2s.npy'
    model_path = '/media/htic/NewVolume1/murali/ecg/codes/memea/v1/models/Sat Jan 20 23:33:47 2018/Epoch24.pt'
#    print os.path.exists(model_path)
#    print model_path
    
    # Parameter settings
    BATCH_SIZE_TEST  = 64
    
    testloader = getTestData(data_path,label_path,BATCH_SIZE_TEST)
    
    # Initialize Model
    ecg = torch.load(model_path)
#    print ecg
    ecg = ecg.cuda()
    
#    print ecg
    
    accuracy_list = []
    
    for step,(x,y) in enumerate(tqdm(testloader)):
        x = Variable(x.cuda())
        y = y.cuda()
        y_predict = ecg(x)
        pred_y = torch.max(y_predict.data,1)[1]
#        print x.shape
#        print y.shape
#        print pred_y.shape
#        rand = np.random.randint(0,64)
#        newval= x.data.cpu()
#        newval = newval.numpy()   
#        newval= newval[0,0,:]
#        print newval.shape                                
#        plt.title("Actual: " +str(y[0]) +" Predicted:" +str(pred_y[0]))
#        plt.plot(newval)
#        plt.show()
        
        accu   = torch.sum(pred_y == y)/float(y.size()[0])
        accuracy_list.append(accu)    
    
    accuracy = np.mean(accuracy_list)
    print accuracy