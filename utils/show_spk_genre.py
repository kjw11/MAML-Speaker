#!/usr/bin/env python
#------------------------------
#	Author: Jiawen Kang
#	Note: Show speaker genre
#-------------------------------

import sys

dir='/work5/cslt/kangjiawen/031320-domain-glzt/MAML-TensorFlow/CNdata/train/spk2utt'
with open(dir, 'r') as f:
    spk2utt = f.readlines()

for line in spk2utt:
    
    name = line.strip().split(' ')[0]
    utts = line.strip().split(' ')[1:]
    print(name)
    print(utts)
    sys.exit()

