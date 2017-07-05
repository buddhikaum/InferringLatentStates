Paper link [http://ieeexplore.ieee.org/document/7952581/].

Use make to build the files.
This work depends on armadillo, scikit-learn and networkx libraries. 

You can run the binary with one command line argument. (E.g. ./var-inf 1)
The argument is a number which can be 1,2 or 3 which indicates, 

1. generating data from the model, 
2. training the model, 
3. inference using trained parameters of the model. 

The script compNetworks.sh will run all the steps including generating random network, 
generating training data and finally the inference on new data set. 

Random network will be written to a file name "random_adj{N}.txt" in the form of 
adjacency list, where N is the number of agents. 

After running compNetwork.sh, compClass.sh can be run to get the AUC-ROC values.

The configuration file (config.txt) includes other required parameters as follows 

line # |  Parameter 

1         Number of agents

2         Time duration

3         Number of latent states

4         Number of samples in Gibbs sampling before writing data for training and testing

5         List of labels of nodes (These labels has to match the adjacency list)

6         Path to write the trained model parameters 

7         Debug mode
