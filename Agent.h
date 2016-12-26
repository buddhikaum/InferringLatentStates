//Agent.h
//#define ARMA_NO_DEBUG
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <iterator>
#include <cmath>

#include <armadillo>
#include <unordered_map>
#include <set>
using namespace std;
using namespace arma;
class Agent
{
	void Initialize(vector<vector<int>>*,unordered_map<int,int>*,int);
	public:
		vector<int> neighIdx;	//neighbors
		static const int T;
		static const int obs_classes;
		static const int K;
		static const bool debug_mode;
		//static int n_obs;
		int nodeId;
		int node_label;
		static const int N;
		static int Ns;
		static bool is_training;
		static bool is_simulating;
		static bool is_infer;
		static string datadir;
		static const string nodes_file;

		vec init_dist;

		mat expression;		//T x obs_classes matrix of measurements
		mat sample_exp;
		vec hidden_state;
		vec hidden_true;
		vector<int> obs_indx;
		bool is_observed;
		//True Parameters 
		mat lambda_true;	//K x obs_classes matrix	
		mat q_prob;
		//cube* theta_true;
		vector<vector<vec> > theta_true;
		//vector<vector<vec> > theta_true_mu;
		vec theta_true_mu;
		mat theta_0_true;
		fstream debugfile;

		mat LD;

		vector<mat> q2_prob;	//This has only T-1 matrices
		Agent(unordered_map<int,int>*,int,vector<vector<int>>*);
		double GetEmission(int,int);
		void SetLD();
		double GetQMarginal(int,int);
		void SetQ2();
};


