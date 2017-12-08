#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys 
from io import open

def split_train_eval(filename, total, ratio=[0.9,0.1]):
    train_num = total * ratio[0]
    eval_num = total * ratio[1]
    train = open(filename+".train", "w", encoding="utf-8")
    eval = open(filename+".eval", "w", encoding="utf-8")

    i = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            sline = line.strip()
            i += 1
            if i< train_num:
                cur_file = train
            elif i-train_num < eval_num:
                cur_file = eval
            cur_file.write(sline + "\n")

if __name__ == "__main__":
    split_train_eval(sys.argv[1], int(sys.argv[2]))


