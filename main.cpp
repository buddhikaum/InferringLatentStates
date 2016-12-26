//#define ARMA_NO_DEBUG
#include <xmmintrin.h>
#include <fenv.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iterator>
#include <armadillo>
#include <unordered_map>
#include <list>
#include <typeinfo>
#include <time.h>
#include "Variational.h"
using namespace std;
using namespace arma;

void ReadData(	string allNodesFile,
				vec &allNodes,
				unordered_map<int,int>*obs_dict)
{
	bool is_read_nodes = allNodes.load(allNodesFile);
	if(!is_read_nodes)
	{
		cerr<<"Failed to read node list "<<allNodesFile<<endl;
		exit(0);
	}
	mat temp_all = allNodes;

	int obsCt = 0;
	for(int i=0;i<(int)allNodes.n_rows;i++)
	{
		(*obs_dict)[allNodes(i)]=obsCt;
		obsCt += 1;
	}
}


void ReadNeighbors(string neighlist_fname,vector<vector<int>>* neigh_list,
					unordered_map<int,int>&obs_dict)
{
	cout<<"Reading adjacencies "<<endl;
	ifstream neigh_list_file(neighlist_fname);
	if(!neigh_list_file)
	{
		cerr<<"Failed to read adjacency file :"<<neighlist_fname<<endl;
		exit(0);
	}
	string neigh_line;
	string user_label;
	int indx=0;
	while(getline(neigh_list_file,neigh_line))
	{
		stringstream linestream(neigh_line);
		vector<int> neighVec;
		getline(linestream,user_label,' ');
		//Omitting first number as this is the node label. Not a neighbor
		indx = obs_dict.at(stoi(user_label));
		while(getline(linestream,user_label,' '))
		{
			neighVec.push_back(stoi(user_label));
		}
		(*neigh_list)[indx] = neighVec;
	}
	//cout<<" neigh list [1] "<<(*neigh_list)[1][3]<<endl;
	//for(vector<int>::iterator k=(*neigh_list)[1].begin();
	//			k!=(*neigh_list)[1].end();k++)
	//{
	//	cout<<k-(*neigh_list)[1].begin()<<" "<<*k<<endl;	
	//}
}


void Supervised(vector<Agent*>&AgentList,unordered_map<int,int>&obs_dict,
				Variational* v)
{
		cout<<"Starting supervised training..."<<endl;
		vector<vector<vector<vec>>> samples_z = vector<vector<vector<vec>>>
										(Agent::N,vector<vector<vec>>
										(Agent::K,vector<vec>
										(Agent::K,vec
										(Agent::obs_classes,fill::zeros))));
		vector<vector<vector<int>>> samples_ab = vector<vector<vector<int>>>
										(Agent::N,vector<vector<int>>
										(Agent::K,vector<int>
										(Agent::K,0)));
		vector<vector<vec>> samples_lmda = vector<vector<vec>>
										(Agent::N,vector<vec>
										(Agent::K,
										 vec(Agent::obs_classes,fill::zeros)));
		double theta_sum =100;
		double lambda_sum=100;
		int while_ct=1;
		mat sum_obs_mat = mat(Agent::T-1,Agent::obs_classes,fill::zeros);

		while(theta_sum>1 or lambda_sum>1)
		{
		
			int n_sample=v->Gibbs(AgentList,obs_dict,samples_z,samples_ab,
														samples_lmda);
			theta_sum=0;
			lambda_sum=0;
			double scale_v = .01;
			double epsilon = scale_v/(1.0+scale_v*while_ct);
			while_ct++;
			double l2_reg = 0;
			for(int kk=0;kk<Agent::N;kk++)
			{
				Agent* current_agent = AgentList[kk];
				mat temp_lambda_mat = current_agent->lambda_true;
				mat q_prob = current_agent->q_prob;
				v->GetZetaMat(AgentList,obs_dict,current_agent,sum_obs_mat,0);
				int nodeId = current_agent->nodeId;
				vec sum_obs(Agent::obs_classes,fill::zeros);
				vector<vector<vec>> theta_grad = vector<vector<vec>>(Agent::K,vector<vec>
											(Agent::K,vec(Agent::obs_classes,fill::zeros)));
				mat theta0_grad = mat(Agent::K,Agent::K,fill::zeros);
				mat grad_lambda = mat(Agent::K,Agent::obs_classes,fill::zeros);
				for(int t=0;t<Agent::T-1;t++)
				{
				
					sum_obs = sum_obs_mat.row(t).t();
					int a = current_agent->hidden_true(t);
					int b = current_agent->hidden_true(t+1);
					theta_grad[a][b] += sum_obs;
					theta0_grad(a,b) += 1;


					grad_lambda.row(a) += current_agent->expression.row(t);
					//Updating lambda using deterministic part of the gradient
					lambda_sum += accu(abs(temp_lambda_mat-current_agent->lambda_true));
					
				}
				grad_lambda.row(current_agent->hidden_true(Agent::T-1)) 
										+= current_agent->expression.row(Agent::T-1);
				for(int a=0;a<Agent::K;a++)
				{
				
					current_agent->lambda_true.row(a) += epsilon*
										(grad_lambda.row(a)-samples_lmda[nodeId][a].t()/n_sample);
					for(int b=0;b<Agent::K;b++)
					{
						current_agent->theta_true[a][b] += epsilon*(theta_grad[a][b]-
															samples_z[nodeId][a][b]/n_sample);
						current_agent->theta_0_true(a,b) += epsilon*(theta0_grad(a,b)-
															samples_ab[nodeId][a][b]/n_sample);
						theta_sum += accu(abs(epsilon*(theta_grad[a][b]-
										samples_z[nodeId][a][b]/n_sample-
											l2_reg*current_agent->theta_true[a][b])))+
									abs(epsilon*(theta0_grad(a,b)-
										samples_ab[nodeId][a][b]/n_sample-
											l2_reg*current_agent->theta_0_true(a,b)));
					}
				}
				lambda_sum += accu(abs(temp_lambda_mat-current_agent->lambda_true));
			}

			if(while_ct%50==0)
			{
				cout<<"theta sum "<<theta_sum<<endl;
				cout<<"lambda sum "<<lambda_sum<<endl;
				cout<<"while ct "<<while_ct<<endl;
			}
		}//End of while

	cout<<"Supervised training completed \n";
	////=======Writing learned model================
	ivec node_list = ivec(Agent::N,fill::zeros);
	node_list.load(Agent::nodes_file);
	for(ivec::iterator node_label=node_list.begin();node_label!=node_list.end();
			node_label++)
	{	
		int agent_idx = obs_dict[*node_label];
		Agent* current_agent = AgentList[agent_idx];
		ofstream theta_vecfile,theta0file,lambdafile;
		theta_vecfile.open("theta_"+to_string(Agent::N)+"_"+
								to_string(*node_label)+"_model.txt");
		vec theta_temp = vec((Agent::K)*(Agent::K)*
						(Agent::obs_classes),fill::zeros);
		int theta_ct =0;
		for(int a=0;a<Agent::K;a++)
		{
			for(int b=0;b<Agent::K;b++)
			{	
				theta_temp.rows(theta_ct*Agent::obs_classes,theta_ct*
										Agent::obs_classes+
										Agent::obs_classes-1)=
											current_agent->theta_true[a][b];
				theta_ct++;
			}			
		}
		theta_temp.print(theta_vecfile);
		theta0file.open("theta0_"+to_string(Agent::N)+"_"+
							to_string(*node_label)+"_model.txt");
		lambdafile.open("lambda_"+to_string(Agent::N)+"_"+
							to_string(*node_label)+"_model.txt");
		current_agent->lambda_true.print(lambdafile);
		current_agent->theta_0_true.print(theta0file);
		theta0file.close();
		theta_vecfile.close();
		lambdafile.close();
	}
}

void GenData(vector<Agent*> &AgentList,
									unordered_map<int,int> &obs_dict
									,Variational *v)
{
	cout<<"Generating data ...\n";
	vector<vector<vector<vec>>> samples_z = vector<vector<vector<vec>>>
									(Agent::N,vector<vector<vec>>
									(Agent::K,vector<vec>
									(Agent::K,vec
									(Agent::obs_classes,fill::zeros))));
	vector<vector<vector<int>>> samples_ab = vector<vector<vector<int>>>
									(Agent::N,vector<vector<int>>
									(Agent::K,vector<int>
									(Agent::K,0)));
	vector<vector<vec>> samples_lmda = vector<vector<vec>>
									(Agent::N,vector<vec>
									(Agent::K,
									 vec(Agent::obs_classes,fill::zeros)));
	v->Gibbs(AgentList,obs_dict,samples_z,samples_ab,samples_lmda);
	cout<<"writing \n";
	//writing samples
	ivec node_list = ivec(Agent::N,fill::zeros);
	int train=0;
	if(Agent::is_training)
		train=2;
	if(Agent::is_simulating)
		train=1;
	node_list.load(Agent::nodes_file);
	for(ivec::iterator node_label=node_list.begin();node_label!=node_list.end();
			node_label++)
	{	
		int agent_idx = obs_dict[*node_label];
		Agent* current_agent = AgentList[agent_idx];
		ofstream measfile,hiddenfile;
		hiddenfile.open("hidden_"+to_string(train)+"_"+to_string(current_agent->node_label)+".txt");
		measfile.open("meas_"+to_string(train)+"_"+to_string(current_agent->node_label)+".txt");
		current_agent->hidden_state.print(hiddenfile);
		current_agent->sample_exp.print(measfile);
		measfile.close();
		hiddenfile.close();
		cout<<"agent "<<*node_label<<" wrote \n";
	}
}

string GetConfig(int i)
{
	fstream conf_file("config.txt",fstream::in);
	string s;
	int ct=1;
	getline(conf_file,s);
	while(ct<i)
	{
		getline(conf_file,s);
		ct++;
	}
	return s;
}

const int Agent::N= stoi(GetConfig(1));
const int Agent::T = stoi(GetConfig(2));
const int Agent::K= stoi(GetConfig(3));
const int Agent::obs_classes = 2;
//C=2
int Variational::test_sample = stoi(GetConfig(4));
const string Agent::nodes_file = GetConfig(5);
static const string neigh_file = GetConfig(6);
string Agent::datadir = GetConfig(7);
const bool Agent::debug_mode = stoi(GetConfig(8));
bool Agent::is_training = false;
bool Agent::is_simulating = false;
bool Agent::is_infer = false;
//int Agent::n_obs = 0;
//Default value should be false 
int main(int argc,char**argv)
{
	if(argc<2)
	{
		cout<<"Not enough arguments \n";
		cout<<	"1. Data generation\n"<<
				"2. Training \n"<<
				"3. Inference\n";
		exit(0);
	}	
	string allNodesFile = Agent::nodes_file;	
	int temp_b = atoi(argv[1]);
	if(temp_b==1)
	{
		//Simulation
		Agent::is_training = false;
		Agent::is_simulating = true;
	}
	else if(temp_b==2)
	{
		Agent::is_training = true;
		Agent::is_simulating = false;
	}
	else if(temp_b==3)
	{
		Agent::is_training = false;
		Agent::is_simulating = false;
		Agent::is_infer = true;
		cout<<"Starting inference.....\n";
	}
	else
	{
		cout<<"Invalid option for training/inference/simulation "<<endl;
		cout<<"Enter option \n"<<
				"1. Data generation\n"<<
				"2. Training \n"<<
				"3. Inference\n";
		exit(0);
	}

	string new_neigh_file = neigh_file+to_string(Agent::N)+".txt";

	cout<<"Time         		:"<<Agent::T<<endl;
	cout<<"# Agents      		:"<<Agent::N<<endl;
	cout<<"# States     		:"<<Agent::K<<endl;
	cout<<"Adjacency file   	:"<<new_neigh_file<<endl;
	cout<<"Node list		:"<<allNodesFile<<endl;
	if(Agent::is_training)
		cout<<"Starting training ............"<<endl;
	if(Agent::is_simulating)
		cout<<"Starting data generation ............"<<endl;
	sleep(1);
	unordered_map<int,int>obs_dict;
	vec *allNodes = new vec(Agent::N);
	vector<Agent*>AgentList(Agent::N);
	vector<vector<int>> neigh_list(Agent::N);
	ReadData(allNodesFile,*allNodes,&obs_dict);
	ReadNeighbors(new_neigh_file,&neigh_list,obs_dict);
	cout<<"Adjacendy file reading completed "<<endl;
	for(int i=0;i<(int)allNodes->n_rows;i++)
	{
		AgentList[i] = new Agent(&obs_dict,(*allNodes)(i),&neigh_list);
	}
	Variational* v = new Variational(AgentList,allNodes,obs_dict);	

	double stKL=0;
	
	vec minVec;
	if(Agent::is_training)
		Supervised(AgentList,obs_dict,v);
	if(Agent::is_simulating)
		GenData(AgentList,obs_dict,v);
	if(Agent::is_infer)
	{
		stKL = v->ComputeKL(AgentList,obs_dict);
		double sum=10;
		while(sum>1e-2)
		{
			sum = v->FullyFactoredUpdate(AgentList,obs_dict);
			stKL = v->ComputeKL(AgentList,obs_dict);
			cout<<"KL "<<stKL<<endl;
			cout<<"q sum "<<sum<<endl;
		}

		v->Generate_ROC(AgentList,obs_dict,allNodes,
						to_string(Agent::N)+".txt");
		cout<<" Roc Computed "<<endl;
	}

	for(int i=0;i<(int)allNodes->n_rows;i++)
	{
		free(AgentList[i]);
	}

}

