#!/usr/bin/env python

import numpy as np
import os

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

sourcedir = "/work5/cslt/kangjiawen/031320-domain-glzt/masf/CNdata/train/txts"
source_list = ['drama', 'entertainment', 'interview', 'live_broadcast', \
                    'play', 'singing', 'vlog', 'movie', 'speech', 'recitation', 'advertisement']
cout = 0
for _ in range(1):
    num_g = len(source_list)
    sampled1, sampled2 = np.random.permutation(num_g)[:2]
    _,l1 =  read_txt_file(sourcedir+source_list[5]+'.txt')
    _,l2 =  read_txt_file(sourcedir+source_list[sampled2]+'.txt')

    names1 = np.unique(l1)
    names2 = np.unique(l2)
    common = [val for val in names1 if val in names2]

    if len(common) <= 8:
        continue;
    else:
        cout = cout + 1
        n1 = 0
        n2 = 0
        print source_list[sampled1], 'spks:', names1
        print source_list[sampled2], 'spks:', names2    
        print source_list[sampled1], source_list[sampled2], "common spks: ", common
        
        for name in common:
            n1 += l1.count(name)
            n2 += l2.count(name)
        print n1
        print n2
#       print n1+n2
print cout 


