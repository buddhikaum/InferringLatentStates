//Variational.h
//#define ARMA_NO_DEBUG
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <cmath>
#include <armadillo>
#include <unordered_map>
#include <set>
#include <list>
#include "Agent.h"
using namespace std;
using namespace arma;

//class Agent;

class Variational
{
	public:
	Variational(unordered_map<int,int>*,mat*);
	Variational(vector<Agent*>&,vec*,unordered_map<int,int>&);
	int T,N,Ns;
	static int test_sample;
	vector<vector<int>> parent_list;
	vector<vector<int>> parent_all;
	void GetZeta(vector<Agent*>&,unordered_map<int,int>&,Agent*,vec &,int);
	void GetZetaMat(vector<Agent*>&,unordered_map<int,int>&,Agent*,mat &,int);
	void GetZetaSamplingMat(vector<Agent*>&,unordered_map<int,int>&,Agent*,mat&);
	vec GetTheta(vector<Agent*>&,unordered_map<int,int>,int,int);
	double ComputeKL(vector<Agent*>&,unordered_map<int,int>&);
	int Gibbs(vector<Agent*>&,unordered_map<int,int>&,vector<vector<vector<vec>>>&,
			vector<vector<vector<int>>>&,vector<vector<vec>>&);
	void GetNeighborThetaMat(int,vector<Agent*>&,int,mat&);
	double FullyFactoredUpdate(vector<Agent*>&,unordered_map<int,int>&);
	void Generate_ROC(vector<Agent*>&,unordered_map<int,int>&,
			vec*,string);
};
