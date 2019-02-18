import os
import glob
import numpy as np
results = []
precisions = []
confusion_matrix = []
save_thresh = []
#1 : abnormal 0 : normal
def caculate_acc (y_true, y_pred):
    
    min_value = np.amin(y_pred)
    max_value = np.amax(y_pred)

    print('min value',min_value)
    print('max value',max_value)
    distance = 0.001
    newthreshold = True
    threshold = min_value 
    
    while newthreshold:
        result = {'precision': 0, 'recall':0 , 'f1_score' : 0}
        
        new_scores = y_pred > threshold 
        newlabels = new_scores * 1
        tmp = {'1-0': 0, '1-1':0, '0-0':0, '0-1':0}
        
        for idx, value in enumerate(y_true):
            if y_true[idx] == 1:
                if newlabels[idx] == 0:
                    tmp['1-0'] +=1
                else:
                    tmp['1-1'] +=1
            if y_true[idx] == 0:
                if newlabels[idx] == 0:
                    tmp['0-0'] += 1
                else:
                    tmp['0-1'] += 1
        conf_matrix = np.array([[tmp['0-0'],tmp['0-1']],[tmp['1-0'],tmp['1-1']]])
        
        result['precision'] += round(tmp['0-0'] / (tmp['1-0'] + tmp['0-0']) , 3)
        result['recall'] += round(tmp['0-0']/(tmp['0-1'] + tmp['0-0'] ),3)
        result['f1_score'] += round( 2/(1/result['precision']+1/result['recall']), 3)
        
        if threshold >= max_value:
            newthreshold = False

        threshold += distance

        if result['precision'] in precisions:
            continue
        if result['precision'] < 0.7:
            continue
        print(result)
        print(conf_matrix)
        save_thresh.append(threshold - 0.01)
        results.append(result)
        precisions.append(result['precision'])
        confusion_matrix.append(conf_matrix)
    
    exit()