#!/usr/bin/env python
#------------------------------
#	Author: Jiawen Kang
#	Note: Make pair data with common genre from CN-Celeb1 dataset, for
#		RobustMAML training. 
#-------------------------------

import numpy as np
import sys
import os
import random


def read_txt_file(file):
    """Read the content of the text file and store it into lists."""
    paths = []
    labels = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(' ')
            paths.append(os.path.join(items[0]))
            labels.append(int(items[1]))
    return paths, labels


def suffule_line(l1, l2):
    """Shuffle 2 list with same shuffle order."""
    lines, l3, l4 = [], [], []
    for i in range(len(l1)):
        lines.append((l1[i], l2[i]))
    random.shuffle(lines)
    for i in range(len(lines)):
        l3.append(lines[i][0])
        l4.append(lines[i][1])
    return l3, l4
 

def random_search(val, list):
    """Return list index with value val"""
    indexs = []
    for i in range(len(list)):
        if list[i] == val:
            indexs.append(i)
    index = random.choice(indexs)
    return int(index)


sourcedir = "/work5/cslt/kangjiawen/031320-domain-glzt/masf/CNdata/train_tdnn/txts"
destdir = "/work5/cslt/kangjiawen/031320-domain-glzt/masf/CNdata/train_tdnn/pairs-ism"
source_list = ['drama', 'entertainment', 'live_broadcast', \
                    'play', 'speech', 'recitation', 'vlog']


for g1 in source_list:
    p1, l1 =  read_txt_file(sourcedir+'/'+g1+ '.txt')
    p1, l1 = suffule_line(p1, l1)
    names1 = np.unique(l1)
    for g2 in source_list:
        if g2 == g1:
            continue
        # make pairs dir
        filedir = destdir+'/'+g1+'-'+g2
        print filedir

        p2, l2 = read_txt_file(sourcedir+'/'+g2+ '.txt')
        p2, l2 = suffule_line(p2, l2)
        names2 = np.unique(l2)
        common = [val for val in names1 if val in names2]
        if len(common) < 8:
            continue
        print "num common spks: ", len(common)

        for i in common:
            if i not in l2:
                print i

        if not os.path.exists(filedir):
            os.makedirs(filedir)    

        # To make sure every list have same label order and maximize utt utilization.
        # Choose longer list as stantard label order
        cout1 = 0
        cout2 = 0
        for i in common:
            cout1 += l1.count(i)  
            cout2 += l2.count(i)
        if cout1 >= cout2:
            genre1, path1, label1 = g1, p1, l1
            genre2, path2, label2 = g2, p2, l2
        else:
            genre1, path1, label1 = g2, p2, l2
            genre2, path2, label2 = g1, p1, l1

        # Make stantard list, and another list follows the same label order.
        with open(filedir+'/'+genre1+'.txt', 'w') as f1:
            cout = 0
            for i,c in enumerate(label1):
                #print i
                if c in common:
                    cout += 1
                    f1.write(path1[i]+' '+str(label1[i])+'\n')   
        print "num common utt", genre1, ': ', cout

        _, std_l =  read_txt_file(filedir+'/'+genre1+ '.txt')

        with open(filedir+'/'+genre2+'.txt', 'w') as f2:
            for i in std_l:
                index = random_search(i, label2)
                f2.write(path2[index]+' '+str(label2[index])+'\n')




