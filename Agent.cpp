//Agent.cpp
//#define ARMA_NO_DEBUG
#include "Agent.h"

using namespace arma;
using namespace std;

Agent::Agent(unordered_map<int,int>* obs_dict,int nodeLabel,
			vector<vector<int>>* neigh_list)
{
	is_observed = true;
	nodeId = obs_dict->at(nodeLabel);
	node_label = nodeLabel;
	if(debug_mode)
	{
		debugfile.open("./debugfiles/"+to_string(nodeLabel)+"debug.db",fstream::out);
		debugfile<<nodeLabel<<" observed"<<endl;
	}
	bool is_meas_file=true;
	if(Agent::is_infer)
		is_meas_file = expression.load("./Testing/meas_1_"+to_string(node_label)+".txt");
	if(Agent::is_training)
		is_meas_file = expression.load("./Training/meas_1_"+to_string(node_label)+".txt");
	expression.print(debugfile,"expression ");
	if(!is_meas_file)
	{
		cerr<<"Failed to read Activity file for agent :"<<node_label<<endl;
	}
	
	Initialize(neigh_list,obs_dict,nodeLabel);
	
}

void Agent::Initialize(vector<vector<int>>*neigh_list,
				unordered_map<int,int>*obs_dict,int nodeLabel)
{
	hidden_state = vec(Agent::T);
	hidden_true = vec(Agent::T);
	if(is_training)
	{
		bool is_hidden_file = hidden_true.load("./Training/hidden_1_"+to_string(node_label)+".txt");
		if(!is_hidden_file)
		{
			cout<<"Failed to read latent state file for agent :"<<node_label<<endl;
			exit(0);
		}
	}
	//Just to generate initial samples for Gibbs 
	init_dist = vec(Agent::K,fill::ones);
	init_dist = init_dist/accu(init_dist);
	q2_prob = vector<mat>(Agent::T-1);
	for(vector<int>::iterator i =(*neigh_list)[nodeId].begin();
							i!=(*neigh_list)[nodeId].end();i++)
	{
		neighIdx.push_back((*i));
		obs_indx.push_back(obs_dict->at(*i));
	}
	debugfile<<"Neigh Idx "<<nodeLabel<<endl;
	ostream_iterator<int> neigh_out(debugfile," ");
	copy(neighIdx.begin(),neighIdx.end(),neigh_out);
	debugfile<<endl;
	debugfile<<"obs indx "<<endl;
	ostream_iterator<int> obs_out(debugfile," ");
	copy(obs_indx.begin(),obs_indx.end(),obs_out);
	debugfile<<endl;
	debugfile<<"unobs labels "<<endl;
	ostream_iterator<int> unobs_out(debugfile," ");
	debugfile<<endl;
	vec theta_temp;
	if(!is_training and !is_simulating)
	{
		theta_temp.load(datadir+"theta_"+to_string(Agent::N)+"_"+
								to_string(node_label)+"_model.txt");
	}
	theta_true = vector<vector<vec> >(Agent::K,vector<vec>(Agent::K,
				vec(Agent::obs_classes,fill::zeros)));
	int theta_ct =0;
	for(int a=0;a<Agent::K;a++)
	{
		for(int b=0;b<Agent::K;b++)
		{	
			if(is_infer)
			{	
				theta_true[a][b] = theta_temp.rows(theta_ct*Agent::obs_classes,theta_ct*
														Agent::obs_classes+
														Agent::obs_classes-1);
				theta_ct++;
			}
			else if(is_simulating)
			{
				if(Agent::K==2)
				{
					theta_true[a][b](0) = -.5*(a+b);
					theta_true[a][b](1) = .5*(a-2*b);
				}
				if(Agent::K==3)
				{
					theta_true[a][b](0) = -.5*((Agent::K)*a+b+1)/(double)
												pow(Agent::K,2);
					theta_true[a][b](1) = .5*((Agent::K)*a+b+1)/(double)
												pow(Agent::K,2);
				}
			}
			else if(is_training)
			{
				//Training no need to initialize	
				//Already initialized
			}
			else
			{
				cerr<<"Incorrect option in parameter initialization"<<endl;
			}
			if(1)
			{
				debugfile<<a<<" "<<b<<endl;
				theta_true[a][b].print(debugfile);
			}
		}			
	}
	if(is_infer)
	{	
		theta_0_true.load(datadir+"theta0_"+to_string(Agent::N)+"_"+
							to_string(node_label)+"_model.txt");
		lambda_true.load(datadir+"lambda_"+to_string(Agent::N)+"_"+
							to_string(node_label)+"_model.txt");
	}
	else if(is_simulating)
	{
		theta_0_true = .1*mat(Agent::K,Agent::K,fill::ones);
		if(Agent::K==2)
			lambda_true = {{-1,1},{1,-1}};
		if(Agent::K==3)
			lambda_true = {{-1,1},{.5,-.5},{1,-1}};
	}
	else if(is_training)
	{
		theta_0_true = mat(Agent::K,Agent::K,fill::zeros);
		lambda_true = mat(Agent::K,Agent::obs_classes,fill::zeros);
	}
	else
	{
		cerr<<"Incorrect option in parameter initialization"<<endl;
	}
	if(debug_mode)
		theta_0_true.print(debugfile,"theta 0");
	if(debug_mode)
		lambda_true.print(debugfile,"lambda_true");
	LD = mat(Agent::T,Agent::K,fill::zeros);

	//Marginal probabilities of latent states
	q_prob = mat(Agent::T,Agent::K);
	for(int t=0;t<Agent::T;t++)
	{
		rowvec rand_t = rowvec(Agent::K,fill::randu);
		q_prob.row(t)	 = rand_t/accu(rand_t);
	}
	
	//Joint of latent states t-1 and t
	for(int t=0;t<Agent::T-1;t++)
	{
		q2_prob[t] = mat(Agent::K,Agent::K,fill::zeros);
	}

	sample_exp = mat(Agent::T,Agent::obs_classes,fill::zeros);
	//======================================================
}

double Agent::GetEmission(int t,int a)
{
	//*******This should be called only if training
	double p=1.0;
	if(is_training or is_simulating)
	{
		for(int nu=0;nu<Agent::obs_classes;nu++)
			p = p*pow(exp(lambda_true(a,nu)),sample_exp(t,nu))/
									tgamma(sample_exp(t,nu)+1);

	}
	else
	{
		cout<<"error... emission called without training\n";
		exit(0);
		p =1;
	}
	return p;
}


void Agent::SetLD()
{
	LD.zeros();
	if(is_observed)
	{
		for(int t=0;t<Agent::T;t++)
		{
			for(int a=0;a<Agent::K;a++)
				for(int nu=0;nu<Agent::obs_classes;nu++)
				{
					LD(t,a) -= expression(t,nu)*lambda_true(a,nu)-
							log(tgamma(expression(t,nu)+1));
				}
		}		
	}
}
double Agent::GetQMarginal(int t,int a)
{
	return q_prob(t,a);
}
void Agent::SetQ2()
{
	for(int t=0;t<Agent::T-1;t++)
	{
		for(int a=0;a<Agent::K;a++)	
		{
			for(int b=0;b<Agent::K;b++)				
			{
				q2_prob[t](a,b) = q_prob(t,a)*q_prob(t+1,b);
			}
		}
		q2_prob[t] = q2_prob[t]/accu(q2_prob[t]);
		if(debug_mode)
			q2_prob[t].print(debugfile,"q2_t at t="+to_string(t));
	}
}
