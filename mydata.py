import numpy as np 
import sys

N = int(sys.argv[2])
T=int(sys.argv[1])
nobs = int(sys.argv[3])
cls_v = sys.argv[4]
H = np.genfromtxt("max_states_"+cls_v+"_"+str(N)+".txt")
#H = np.genfromtxt("max_states_2010var.txt")
labl = np.genfromtxt("int.txt")

newMat = np.zeros(N*T)
for i in range(0,N):

	#newMat[H[i,0],:] = H[i,1:21]
	#np.vstack((newMat,H[i,1:21].reshape(20,1)))
	newMat[i*T:(i+1)*T] = H[i,1:T+1]
	#print H[i,0]

np.savetxt("my_pvals.txt",newMat)
