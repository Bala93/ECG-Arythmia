
import sys
import glob
import numpy as np

### Code to read the text files and report the average accuracy"

args = sys.argv[1:]
log_path = './' + args[0] + '*.txt'
log_files = glob.glob(log_path)
#print log_files

accuracy = []

for each in log_files:
    f = open(each)
    lines = f.readlines()
    max_acc = 0.0
    
    for i in lines:
        acc = float(i.split('|')[-1].split(':')[-1].strip())
        if acc > max_acc:
            max_acc = acc
            
    accuracy.append(max_acc)
    
final_accuracy = np.mean(accuracy)
print "List of accuracies"
print accuracy
print "Average of accuracies"
print final_accuracy
