#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import open

def split_train_dev(filename, total, ratio=[0.9,0.1]):
    train_num = total * ratio[0]
    dev_num = total * ratio[1]
    train = open(filename+".train", "w", encoding="utf-8")
    dev = open(filename+".dev", "w", encoding="utf-8")

    i = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            sline = line.strip()
            i += 1
            if i< train_num:
                cur_file = train
            elif i-train_num < dev_num:
                cur_file = dev
            cur_file.write(sline + "\n")

if __name__ == "__main__":
    split_train_dev("msr_training.utf8", 86924)

