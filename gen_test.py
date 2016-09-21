# -*- coding: utf-8 -*-
"""
Created on Mon May 09 12:11:55 2016

@author: LHD
"""
from sklearn.datasets import make_moons
def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in xrange(0, len(l), group_size):
        yield l[i:i+group_size]
        
data, label = make_moons(n_samples=6, noise=0.20)
s = group_list(data,6)
s.
