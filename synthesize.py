import os
import sys
import numpy as np
import pandas as pd
import pdb

def sys_self(df):
    L = df.shape[0]
    p1 = np.array(df.iloc[:,[1, 2]])
    p2 = np.array(df.iloc[:,[3, 4]])
    
    accu = []
    for w in range(0, 100):
        w = float(w) / 100
        p = w * p1 + (1 - w) * p2
        accu_tmp = float(sum(p.argmax(1) == df.iloc[:, 5])) / L
        accu.append(accu_tmp)
        
    #print(max(accu))
    
    return max(accu)
    

def sys_cross(df1, df2):
    L = df1.shape[0]
    p1 = np.array(df1.iloc[:,[2, 4]])
    p2 = np.array(df2.iloc[:,[2, 4]])
    
    accu = []
    num_cat = 500
    for w in range(0, num_cat):
        w = float(w) / num_cat
        p = w * p1 + (1 - w) * p2
        accu_tmp = float(sum(p.argmax(1) == df1.iloc[:, 5])) / L
        accu.append(accu_tmp)
        
    #print(max(accu))
    return max(accu)


path = '/users4/ldu/GraphBert/records_sct/predicts/'
os.chdir(path)
ls = os.listdir(path)

accus = [0]
s1 = []
s2 = []
for l1 in ls:
    for l2 in ls:         
        f1 = pd.read_csv(l1)
        f2 = pd.read_csv(l2)
        
        #sys_self(f1)
        #sys_self(f2)
        accu_tmp = sys_cross(f1, f2)
        
        if accu_tmp > max(accus):
            accus.append(accu_tmp)
            print(accu_tmp)
            print(l1)
            print(l2)

      

'''
        if (l1 not in s1) or (l2 not in s2):
            s1.append(l1)
            s2.append(l2)
'''
