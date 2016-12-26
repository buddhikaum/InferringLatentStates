import numpy as np 
import sys
from sklearn import metrics
#import matplotlib.pyplot as plt

cls_v = int(sys.argv[1])
np.set_printoptions(threshold='nan')
true_states1 = np.genfromtxt("true_states.txt")
true_states = np.zeros((len(true_states1),1))
true_states[np.where(true_states1==cls_v)] = 1
est_fact = np.genfromtxt("my_pfact.txt")


print metrics.roc_auc_score(true_states,est_fact)
