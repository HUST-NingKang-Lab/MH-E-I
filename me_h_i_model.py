#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------
# description
# ---------------------------------------------------------------

"""
me_h_i_module.py
============
Please type "./me_h_i_module.py -h" for usage help
    
Author:
    Li Hongjun

Description:
    This script is used for train and test mei model.
    ME(H)I model:
    mei = (∑(i=1-n){ABU_r}(Si))/(∑(j=1-n){ABU_r}(Sj))

Reurirements:
    Python packages: argparse, time, sklearn, scipy, matplotlib
"""

# ---------------------------------------------------------------
# import
# ---------------------------------------------------------------

import argparse
import time
#import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

# ---------------------------------------------------------------
# function definition
# ---------------------------------------------------------------

def parse_args():
    """ master argument parser """
    parser = argparse.ArgumentParser(
        description = """        
        This script is used for train and test mei model.
        Input features to the model from file and data is formatted as below (separated by tab):
        group  f1_B            f1_K            f-1_T
        1      0.017326722     0.257304377     0.0003404
        1      0.004271432     0.12957425      0.000656918
        1      0.0015858       0.085703255     0.015039975
        -1     0.000692197     0.102700699     0.039183578
        -1     0.003202926     0.135973633     0.027555026
        -1     0.004132046     0.114394002     0.011587037
        """,
        #epilog="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    parser.add_argument(
        '-f', '--feature_input',
        type=str,
        required=True,
        help="""
        Input features to the model from file.
        """,
        )
    parser.add_argument(
        '-t', '--test_input',
        type=str,
        required=True,
        help="""
        Input test dataset to the model from file.
        """,
        )
    parser.add_argument(
        '-g', '--group_name',
        type=str,
        required=False,
        default='f1',
        help="""
        Input group name to the model from file(default: f1).
        """,
    )
    args = parser.parse_args()
    return args

def create_dataset(_file, _group):
    """ obtain feature data from feature_input file """
    dataset = []
    label = []
    count_f1 = _file.readline().count(_group)
    for _line in _file.readlines()[0:]:
        dataset.append([float(ele) for ele in _line.split()[1:]])
        label.append(int(_line.split()[0]))
    return dataset, label, count_f1

def mei(feature_list, count_f1):
    """ cal mei of each sample """
    sum_f1 = sum(feature_list[:count_f1])
    sum_f2 = sum(feature_list[count_f1:])
    if sum_f2 > 0:
        return round(sum_f1/sum_f2, 9)
    else:
        return 0
def sigmoid(X):
    """ sigmoid function """
    if X>0:
        return 1
    elif X<0:
        return -1
    else:
        return 0

def train_threshold(th_max, th_min, mei_list, label):
    """ 
    through length of step set, try the performance of each value as threshold,
    finally return the relatively best threshold.
    """
    
    delta = 1.0
    th = th_min    # th: element of list of candidate thresholds
    ths = [th_min + (th_max - th_min) * n/1000 for n in range(1001)]
    print "Train start from threshold %f and Step length is set to %f\n" % (
        th, (th_max - th_min)/1000)
    # list of candidate thresholds
    for th in ths:
        tp = tn = fp = fn = acc = 0.0
        for i in range(len(mei_list)):
            if sigmoid(mei_list[i] - th) == label[i]:
                acc += 1
                if sigmoid(mei_list[i] - th) == 1:
                    tp += 1.0
                else:
                    tn += 1.0
            else:
                if sigmoid(mei_list[i] - th) == 1:
                    fp += 1.0
                else:
                    fn += 1.0
        Sp = tn / (tn + fp)
        Sn = tp / (tp + fn)
        accuarcy = acc/len(mei_list)
        if abs(Sn - Sp) < delta:
            delta = abs(Sn - Sp)
            threshold = th
            threshold_accuracy = accuarcy
            print "Threshold updated to %f with Sn: %f and Sp: %f" % (
                threshold, Sn, Sp)

    return threshold, threshold_accuracy

def cal_accuracy(mei_list, label, threshold):
    """ calculate accuracy based on the threshold """
    acc_count = 0.0
    for i in range(len(mei_list)):
        if sigmoid(mei_list[i] - threshold) == label[i]:
            acc_count += 1.0
        else:
            continue
    return acc_count/len(mei_list)

def cal_tpr_tnr(mei_list, label, threshold):
    """
    calculate sensitivity and specificity, 
    sensitivity (tpr) = tp / (tp + fn), 
    specificity (tnr) = tn / (tn + fp)
    """
    tp = tn = fp = fn = 0.0
    for i in range(len(mei_list)):
        if sigmoid(mei_list[i] - threshold) == label[i]:
            if sigmoid(mei_list[i] - threshold) == 1:
                tp += 1.0
            else:
                tn += 1.0
        else:
            if sigmoid(mei_list[i] - threshold) == 1:
                fp += 1.0
            else:
                fn += 1.0
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return tpr, tnr

def roc_curve(th_max, th_min, mei_list, label):
    """ plot roc curve of the model """
    tp = tn = fp = fn = 0.0
    fprs = []
    tprs = []
    
    th = th_min    # th: element of list of candidate thresholds
    ths = [th_min + (th_max - th_min) *n/1000 for n in range(1001)]   
    # list of candidate thresholds
    for th in ths:
        for i in range(len(mei_list)):
            if sigmoid(mei_list[i] - th) == label[i]:
                if sigmoid(mei_list[i] - th) == 1:
                    tp += 1.0
                else:
                    tn += 1.0
            else:
                if sigmoid(mei_list[i] - th) == 1:
                    fp += 1.0
                else:
                    fn += 1.0
        fprs.append(fp / (fp + tn))
        tprs.append(tp / (tp + fn))
    
    return fprs, tprs

def plot_roc_curve(fprs, tprs):
    """ plot roc curve of the model """
    plt.plot(fprs, tprs)
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

def cal_precision_recall_f1(mei_list, label, threshold):
    """
    calculate precision and recall, 
    precision = tp / (tp + fp), 
    recall = tp / (tp + fn)
    """
    tp = tn = fp = fn = 0.0
    for i in range(len(mei_list)):
        if sigmoid(mei_list[i] - threshold) == label[i]:
            if sigmoid(mei_list[i] - threshold) == 1:
                tp += 1.0
            else:
                tn += 1.0
        else:
            if sigmoid(mei_list[i] - threshold) == 1:
                fp += 1.0
            else:
                fn += 1.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*tp/(len(mei_list) + tp - tn)
    return precision, recall, f1

def classify_all(mei_list, threshold):
    """ get prediction result of test data """
    pred_result = []
    for ele in mei_list:
        pred_result.append(sigmoid(ele - threshold))
    return pred_result

def plot_precision_recall_curve(mei_list, label):
    """ plot precision recall curve of the model """
    precision, recall = \
    precision_recall_curve(label, mei_list)[:-1]
    plt.plot(recall, precision)
    plt.title('P-R Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------

def train(args):
    """ train """
    with open(args.feature_input, 'r') as feature_input:
        dataset,label,count_f1= create_dataset(feature_input, args.group_name)
    
    # save mei scores of all train_data
    mei_list = []
    for i in range(len(dataset)):
        mei_list.append(mei(dataset[i], count_f1))
    
    with open("train_mei_list_for_roc_r.txt", 'w+') as output:
        output.write("status\tmodel_score\n")
        for i in range(0, len(mei_list)):
            output.write(
                str(label[i]) + "\t" + str(mei_list[i]) + "\n"
                )

    # the max and min of mei values of all samples.
    th_max = max(mei_list)
    th_min = min(mei_list)
    
    # get final threshold based on accuracy percents and cal tpr and tnr
    threshold, threshold_accuracy= train_threshold(
        th_max, th_min, mei_list, label)
    print "\nThreshold trained: %f\n\nResults based on training dataset:\nAccuracy percent: %f%%" % \
    (threshold, threshold_accuracy * 100)
    
    tpr,tnr = cal_tpr_tnr(mei_list, label, threshold)
    print "Sensitivity: %f\tSpecificity: %f" % (tpr, tnr)
    
    ### auc score of the model
    auc_score = roc_auc_score(label, mei_list)
    print "Model AUC: %f" % auc_score
    
    ### calculate precision and recall
    precision, recall, f1 = cal_precision_recall_f1(mei_list, label, threshold)
    print "Precision: %f\tRecall: %f\nf1: %f" % (precision, recall, f1)
    
    # plot roc curve, based on roc, get final threshold
    #===========================================================================
    # fprs, tprs = roc_curve(th_max, th_min, mei_list, label)
    # plot_roc_curve(fprs, tprs)
    # plot_precision_recall_curve(mei_list, label)
    #===========================================================================
    return threshold
    
def test(args, threshold_trained):
    """ test """
    with open(args.test_input, 'r') as test_input:
        dataset,label,count_f1= create_dataset(test_input, args.group_name)
    
    mei_list = []
    for i in range(len(dataset)):
        mei_list.append(mei(dataset[i], count_f1))

    with open("test_mei_list_for_roc_r.txt", 'w+') as output:
        output.write("status\tmodel_score\n")
        for i in range(0, len(mei_list)):
            output.write(
                str(label[i]) + "\t" + str(mei_list[i]) + "\n"
                )
    
    print "Model threshold: %f\n" % threshold_trained
    
    print "Prediction result of test data is:", classify_all(
        mei_list, threshold_trained)
    test_accuracy = cal_accuracy(mei_list, label, threshold_trained)
    print "Accuracy percent: %f%%" % (test_accuracy * 100)
    
    tpr,tnr = cal_tpr_tnr(mei_list, label, threshold_trained)
    print "Sensitivity: %f\tSpecificity: %f" % (tpr, tnr)
    
    ### auc score of the model
    auc_score = roc_auc_score(label, mei_list)
    print "Model AUC: %f" % auc_score
    
    ### calculate precision and recall
    precision, recall, f1 = cal_precision_recall_f1(
        mei_list, label, threshold_trained)
    print "Precision: %f\tRecall: %f\nf1: %f" % (precision, recall, f1)
    
    #===========================================================================
    # th_max = max(mei_list)
    # th_min = min(mei_list)
    # fprs, tprs = roc_curve(th_max, th_min, mei_list, label)
    # plot_roc_curve(fprs, tprs)
    # plot_precision_recall_curve(mei_list, label)
    #===========================================================================

def main():
    """ main """
    args = parse_args()
    print "Training ......\n"
    threshold_trained = train(args)
    print "\nTesting ......\n"
    test(args, threshold_trained)
    
if __name__ == "__main__":
    start = time.clock()
    main()
    end = time.clock()
    print('\nFinish all in %s seconds' % str(end - start))