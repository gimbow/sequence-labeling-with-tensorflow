#!/usr/bin/env python
# coding=utf8

import sys
import numpy as np
from collections import OrderedDict

def main(vocab_file, pretrain_file):  
    d = OrderedDict()
    with open(vocab_file, "r") as fp:
        for line in fp:
            line = line.strip()
            data = line.split()
            d[data[0]] = data[1]
    with open(pretrain_file, "r") as fp:
        for line in fp:
            line = line.strip()
            data = line.split()
            if data[0] in d:
                continue
            else:
                d[data[0]] = "1"
    for key in d:
        print "%s\t%s" % (key, d[key])
    
    return 0

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

