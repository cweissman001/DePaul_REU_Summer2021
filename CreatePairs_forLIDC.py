# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:12:54 2021
Pair creation for spiculation
@author: weissmce
"""


num_classes = 2
def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    ##length of the shortest index in digit indices minus 1
    for d in range(num_classes):
    ##looping through all classes
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            #print(z1, z2, labels)
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            #print(z1, z2, labels)
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
            #print(np.array(labels))
    return np.array(pairs), np.array(labels)


def createPairs(DATA):
    pairs = []
    labels = []
    n = DATA.length
    for x in range(n):
        if DATA[x]['Agglomeration'] == 1:
            for 
            
            
            
            
"""
from itertools import permutations
for i in permutations(your_list, 3):
"""