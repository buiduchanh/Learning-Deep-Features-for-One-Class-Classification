import os
import glob
import numpy as np
import csv

results = []
precisions = []
confusion_matrix = []
save_thresh = []
#1 : abnormal 0 : normal
def caculate_acc (y_true, y_pred, path):
    
    with open('testing_result_vgg_cp1_newthod.csv', mode='w', newline='') as csv_file:
        fieldnames = ['Image', 'Real Label','Number_Label', 'Scores']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        writer.writeheader()
        for index, path_ in enumerate(path):
            basepath = os.path.basename(path_).strip()
            number_label = y_true[index]
            print(number_label, type(number_label))
            if number_label == int(0):
                label_ = 'OK'
            else :
                label_ = 'NG'
            writer.writerow({'Image': basepath, 'Real Label': label_, 
                            'Number_Label': int(number_label),'Scores': round(y_pred[index],3)})

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
        print(result, threshold - 0.01)
        print(conf_matrix)
        save_thresh.append(threshold - 0.01)
        results.append(result)
        precisions.append(result['precision'])
        confusion_matrix.append(conf_matrix)
    
    return precisions, save_thresh, results