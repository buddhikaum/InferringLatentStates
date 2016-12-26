//Variational.cpp
//#define ARMA_NO_DEBUG
#include <iostream>
#include <string>
#include <sstream>
#include <time.h>
//#include <cstdlib>
#include "Variational.h"
using namespace std;
using namespace arma;

Variational::Variational(vector<Agent*>&agent_list,
			vec *allNodes,unordered_map<int,int>&obs_dict)
{
	parent_all = vector<vector<int>>(Agent::N);
	for(int n=0;n<Agent::N;n++)
	{
		for(vector<int>::iterator all_idx=agent_list[n]->neighIdx.begin();
				all_idx!=agent_list[n]->neighIdx.end();all_idx++)
		{
			parent_all[obs_dict.at(*all_idx)].push_back(n);
		}
	}
}

void Variational::GetZeta(vector<Agent*>& agent_list,
							unordered_map<int,int>& obs_dict,
							Agent* current_agent,
							vec &sum_obs,int t)
{
	sum_obs.zeros();
	for(vector<int>::iterator obs_node=current_agent->obs_indx.begin();
				obs_node!=current_agent->obs_indx.end();obs_node++)
	{
		sum_obs += agent_list[*obs_node]->expression.row(t).t();
	}
	if(current_agent->neighIdx.size()>0)
		sum_obs = sum_obs/(current_agent->neighIdx.size());
	if(current_agent->debug_mode)
	{
		sum_obs.print(current_agent->debugfile,"sum of obs at GetZeta");
	}
}

void Variational::GetZetaMat(vector<Agent*>& agent_list,
							unordered_map<int,int>& obs_dict,
							Agent* current_agent,
							mat &sum_obs,int t)
{
	sum_obs.zeros();
	for(vector<int>::iterator obs_node=current_agent->obs_indx.begin();
				obs_node!=current_agent->obs_indx.end();obs_node++)
	{
		sum_obs += agent_list[*obs_node]->expression.rows(0,Agent::T-2);
	}
	if(current_agent->neighIdx.size()>0)
		sum_obs = sum_obs/(current_agent->neighIdx.size());
	if(current_agent->debug_mode)
	{
		sum_obs.print(current_agent->debugfile,"sum of obs at GetZeta");
	}
}

void Variational::GetZetaSamplingMat(vector<Agent*>& agent_list,
								unordered_map<int,int>& obs_dict,
								Agent* current_agent,
								mat &sum_obs_mat)
{
	sum_obs_mat.zeros();
	for(vector<int>::iterator neigh_it=current_agent->neighIdx.begin();
			neigh_it!=current_agent->neighIdx.end();neigh_it++)
	{
		sum_obs_mat += agent_list[obs_dict.at(*neigh_it)]->sample_exp;
	}
	if(current_agent->neighIdx.size()>0)
	{
		sum_obs_mat = sum_obs_mat/current_agent->neighIdx.size();
	}
}


double Variational::ComputeKL(vector<Agent*>& agent_list,
							unordered_map<int,int>& obs_dict)
{
	//This method computes KL divergence as well as 
	//update statistics 
	Agent *current_agent;
	double KL=0;
	double full_kl = 0;
	double exp_p;
	vec sum_obs(Agent::obs_classes,fill::zeros);
	for(int kk=0;kk<Agent::N;kk++)
	{
		current_agent = agent_list[kk];
		current_agent->SetQ2();
		current_agent->SetLD();
	}
	for(int kk=0;kk<Agent::N;kk++)
	{
		current_agent = agent_list[kk];
		KL = 0;
		if(current_agent->debug_mode)
			current_agent->debugfile<<"Starting KL for "<<current_agent->nodeId<<endl;

		for(int a=0;a<Agent::K;a++)
		{
			KL += current_agent->GetQMarginal(0,a)*current_agent->LD(0,a);
			//Fully
			if(current_agent->GetQMarginal(0,a)>0)
				KL += current_agent->GetQMarginal(0,a)*log(current_agent->GetQMarginal(0,a));
		}
		for(int t=1;t<Agent::T;t++)
		{
			for(int a=0;a<Agent::K;a++)	
			{
				if(current_agent->GetQMarginal(t,a)>0)
					KL += current_agent->GetQMarginal(t,a)*log(current_agent->GetQMarginal(t,a));
				for(int b=0;b<Agent::K;b++)	
				{
					GetZeta(agent_list,obs_dict,current_agent,sum_obs,t-1);
					exp_p = dot(sum_obs,current_agent->theta_true[a][b])+
								current_agent->theta_0_true(a,b);
					KL -= current_agent->q2_prob[t-1](a,b)*exp_p;
				}
				KL += current_agent->GetQMarginal(t,a)*current_agent->LD(t,a);
			}
		}
		if(current_agent->debug_mode)
			current_agent->debugfile<<"AgentID "<<current_agent->nodeId<<
											" KL done."<<KL<<endl;
		full_kl = full_kl+KL;
	}
	return full_kl;
}

double Variational::FullyFactoredUpdate(vector<Agent*>&agent_list
				,unordered_map<int,int>&obs_dict)
{
	double diff_sum=0;
	for(vector<Agent*>::iterator current_agent_ite=agent_list.begin();
			current_agent_ite!=agent_list.end();current_agent_ite++)
	{
		Agent* current_agent = *current_agent_ite;
		mat approx_q = current_agent->q_prob;
		vec sum_obs(Agent::obs_classes,fill::zeros);
		GetZeta(agent_list,obs_dict,current_agent,sum_obs,0);
		rowvec temp_q = rowvec(Agent::K,fill::zeros);
		rowvec pre_r = approx_q.row(0);
		for(int a=0;a<Agent::K;a++)
		{
			double exp_p=0;
			for(int b=0;b<Agent::K;b++)	
			{
			
				exp_p += approx_q(1,b)*(dot(sum_obs,
							current_agent->theta_true[a][b])+
							current_agent->theta_0_true(a,b));

			}
			exp_p -= current_agent->LD(0,a);
			temp_q(a) = exp_p;
		}
		rowvec exp_temp1 = exp(temp_q);
		approx_q.row(0) = exp(temp_q)/accu(exp(temp_q));
		diff_sum += accu(abs(pre_r-approx_q.row(0)));
		for(int t=1;t<Agent::T-1;t++)	
		{
			vec sum_obs_nxt(Agent::obs_classes,fill::zeros);
			GetZeta(agent_list,obs_dict,current_agent,sum_obs_nxt,t);
			GetZeta(agent_list,obs_dict,current_agent,sum_obs,t-1);
			pre_r = approx_q.row(t);
			for(int a=0;a<Agent::K;a++)
			{
				//this is for updating x_t t=0 upto t-1
				double exp_p=0;
				for(int b=0;b<Agent::K;b++)	
				{
									
					exp_p += approx_q(t+1,b)*(dot(sum_obs_nxt,
								current_agent->theta_true[a][b])+
								current_agent->theta_0_true(a,b));
					exp_p += approx_q(t-1,b)*(dot(sum_obs,
								current_agent->theta_true[b][a])+
								current_agent->theta_0_true(b,a));
				}
				exp_p -= current_agent->LD(t,a);
				temp_q(a) = exp_p;
			}
			rowvec exp_temp = exp(temp_q);
			temp_q = temp_q-temp_q.max();
			approx_q.row(t) = exp(temp_q)/accu(exp(temp_q));
			diff_sum += accu(abs(pre_r-approx_q.row(t)));
		}
		GetZeta(agent_list,obs_dict,current_agent,sum_obs,Agent::T-2);
		pre_r = approx_q.row(Agent::T-1);
		for(int a=0;a<Agent::K;a++)
		{
			double exp_p=0;
			for(int b=0;b<Agent::K;b++)	
			{
			
				exp_p += approx_q(Agent::T-2,b)*(dot(sum_obs,
							current_agent->theta_true[b][a])+
							current_agent->theta_0_true(b,a));

			}
			exp_p -= current_agent->LD(Agent::T-1,a);
			temp_q(a) = exp_p;
		}
		approx_q.row(Agent::T-1) = exp(temp_q)/accu(exp(temp_q));
		diff_sum += accu(abs(pre_r-approx_q.row(Agent::T-1)));
		current_agent->q_prob = approx_q;
	}
	return diff_sum;
}

int Variational::Gibbs(vector<Agent*> &agent_list,
								unordered_map<int,int>& obs_dict,
								vector<vector<vector<vec>>>& samples_z,
								vector<vector<vector<int>>>& samples_ab,
								vector<vector<vec>>& samples_lmda)
{

	Agent *current_agent;
	mat state_sample = mat(Agent::N,Agent::T);
	random_device rd;
	default_random_engine p_gen(rd()); 
	int temp_state=0;
	std::fill(samples_z.begin(),samples_z.end(),
				vector<vector<vec>>(Agent::K,vector<vec>(Agent::K,vec(Agent::obs_classes,
							fill::zeros))));
	std::fill(samples_ab.begin(),samples_ab.end(),
				vector<vector<int>>(Agent::K,vector<int>(Agent::K,0)));
	std::fill(samples_lmda.begin(),samples_lmda.end(),
				vector<vec>(Agent::K,vec(Agent::obs_classes,
							fill::zeros)));
	//Initialization
	for(vector<Agent*>::iterator current_agent_ite=agent_list.begin();
						current_agent_ite!=agent_list.end();current_agent_ite++)
	{
		current_agent = *current_agent_ite;
		discrete_distribution<int> mult_dist (current_agent->init_dist.begin()
										,current_agent->init_dist.end());
		temp_state = mult_dist(p_gen);
		for(int t=1;t<Agent::T;t++)	
		{
			for(int nu=0;nu<Agent::obs_classes;nu++)
			{
				poisson_distribution<int> poi_dist(exp(current_agent->lambda_true
													(temp_state,nu)));
				current_agent->sample_exp(t-1,nu)= poi_dist(p_gen);
			}
			current_agent->hidden_state(t-1) = temp_state;
			vec temp_prob = vec(Agent::K,fill::zeros);
			for(int i=0;i<Agent::K;i++)
			{
				temp_prob(i) = exp(current_agent->theta_0_true(temp_state,i));
			}
			temp_prob = temp_prob/accu(temp_prob);
			discrete_distribution<int> tx_dist (temp_prob.begin(),temp_prob.end());
			temp_state = tx_dist(p_gen);
		}
		current_agent->hidden_state(Agent::T-1) = temp_state;
		for(int nu=0;nu<Agent::obs_classes;nu++)
		{
			poisson_distribution<int> poi_dist(exp(current_agent->lambda_true
												(temp_state,nu)));
			current_agent->sample_exp(Agent::T-1,nu)= poi_dist(p_gen);
		}
	}
	//Initial sampling over 
	
	vec temp_prob = vec(Agent::K,fill::zeros);
	vec theta_sum = vec(Agent::obs_classes,fill::zeros);
	vec sum_obs_pre = vec(Agent::obs_classes,fill::zeros);
	vec sum_obs_nxt = vec(Agent::obs_classes,fill::zeros);
	discrete_distribution<int> tx_dist;
	int nxt_st,pre_st;
	int n_sample = test_sample;
	//For training small number of samples were used 
	//to compute stochastic gradients
	if(Agent::is_training)
		n_sample = 20;
	int sample_ct = 0;
	for(int s=0;s<n_sample;s++)
	{
		for(vector<Agent*>::iterator current_agent_ite=agent_list.begin();
							current_agent_ite!=agent_list.end();current_agent_ite++)
		{
			current_agent = *current_agent_ite;

			mat sum_obs_mat = mat(Agent::T,Agent::obs_classes,fill::zeros);
			mat theta_sum_mat = mat(Agent::obs_classes,Agent::T-1,fill::zeros);
			GetZetaSamplingMat(agent_list,obs_dict,current_agent,sum_obs_mat);
			GetNeighborThetaMat(0,agent_list,current_agent->nodeId,theta_sum_mat);

			nxt_st = current_agent->hidden_state(1);
			sum_obs_nxt = sum_obs_mat.row(0).t();
			for(int i=0;i<Agent::K;i++)
			{
				temp_prob(i) = exp(current_agent->theta_0_true(i,nxt_st)+
									dot(current_agent->theta_true[i][nxt_st]
															,sum_obs_nxt))*
									current_agent->init_dist(i)*
									current_agent->GetEmission(0,i);
			}
			temp_prob = temp_prob/accu(temp_prob);
			tx_dist =discrete_distribution<int>(temp_prob.begin(),temp_prob.end());
			current_agent->hidden_state(0) = tx_dist(p_gen);
			pre_st = current_agent->hidden_state(0);
			//Initial hidden state sampled 
			for(int t=1;t<Agent::T-1;t++)	
			{

				theta_sum = theta_sum_mat.col(t-1);

				for(int nu=0;nu<Agent::obs_classes;nu++)
				{
					poisson_distribution<int> poi_dist(exp(current_agent->lambda_true
														(pre_st,nu))*theta_sum(nu));
					current_agent->sample_exp(t-1,nu)= poi_dist(p_gen);
				}
				//End update measurement Going to next hidden state

				sum_obs_pre = sum_obs_mat.row(t-1).t();
				sum_obs_nxt = sum_obs_mat.row(t).t();


				nxt_st = current_agent->hidden_state(t+1);
				for(int i=0;i<Agent::K;i++)
				{
					temp_prob(i) = exp(current_agent->theta_0_true(pre_st,i)+
								dot(current_agent->theta_true[pre_st][i],sum_obs_pre))*
								exp(current_agent->theta_0_true(i,nxt_st)+
								dot(current_agent->theta_true[i][nxt_st],sum_obs_nxt))*
										current_agent->GetEmission(t,i);
				}
				temp_prob = temp_prob/accu(temp_prob);
				tx_dist =discrete_distribution<int>(temp_prob.begin(),temp_prob.end());
				pre_st = tx_dist(p_gen);
				current_agent->hidden_state(t) = pre_st;
			}
			//At this point hidden states upto T-2 and meas upto T-3 updated
			//theta_sum=GetNeighborTheta(Agent::T-2,agent_list,current_agent->nodeId);
			theta_sum = theta_sum_mat.col(Agent::T-2);
			for(int nu=0;nu<Agent::obs_classes;nu++)
			{
				poisson_distribution<int> poi_dist(exp(current_agent->lambda_true
													(pre_st,nu))*theta_sum(nu));
				current_agent->sample_exp(Agent::T-2,nu)= poi_dist(p_gen);
			}
			sum_obs_pre = sum_obs_mat.row(Agent::T-2).t();
			for(int i=0;i<Agent::K;i++)
			{
				temp_prob(i) = exp(current_agent->theta_0_true(pre_st,i)+
							dot(current_agent->theta_true[pre_st][i],sum_obs_pre))*
									current_agent->GetEmission(Agent::T-1,i);
			}
			temp_prob = temp_prob/accu(temp_prob);
			tx_dist = discrete_distribution<int>(temp_prob.begin(),temp_prob.end());
			pre_st = tx_dist(p_gen);
			current_agent->hidden_state(Agent::T-1) = pre_st;
			for(int nu=0;nu<Agent::obs_classes;nu++)
			{
				poisson_distribution<int> poi_dist(exp(current_agent->lambda_true
																(pre_st,nu)));
				current_agent->sample_exp(Agent::T-1,nu)= poi_dist(p_gen);
			}
		}
		if(s>n_sample*.1)
		{
			sample_ct++;
			mat sum_obs_mat = mat(Agent::T,Agent::obs_classes,fill::zeros);
			for(vector<Agent*>::iterator current_agent_ite=agent_list.begin();
								current_agent_ite!=agent_list.end();current_agent_ite++)
			{
				current_agent = *current_agent_ite;
				int nodeId = current_agent->nodeId;
				pre_st = current_agent->hidden_state(0);
				samples_lmda[nodeId][pre_st] += current_agent->sample_exp.row(0).t();
				GetZetaSamplingMat(agent_list,obs_dict,current_agent,sum_obs_mat);
				if(any(vectorise(sum_obs_mat)<0))
				{
					sum_obs_mat.print("sum obs mat");
					cout<<"label "<<current_agent->node_label<<endl;
					exit(0);
				}
				for(int t=1;t<Agent::T;t++)
				{
					sum_obs_pre = sum_obs_mat.row(t-1).t();
					//Collecting z_{t-1}^j \in N(i) samples 
					nxt_st = current_agent->hidden_state(t);
					samples_z[nodeId][pre_st][nxt_st] += sum_obs_pre;
					samples_ab[nodeId][pre_st][nxt_st] += 1; 
					samples_lmda[nodeId][nxt_st] += current_agent->sample_exp.row(t).t();
					pre_st = nxt_st;
				}
			}
		}
		//End of Agents----------------------------
	}
	//End of s_th Sample---------------------------------------
	return sample_ct;
}


void Variational::GetNeighborThetaMat(int tt,vector<Agent*>& agent_list,
							int list_idx,
							mat &sum_theta_mat)
{
	Agent* current_agent;
	vec sum_theta = vec(Agent::obs_classes,fill::zeros);
	sum_theta_mat.zeros();
	for(int t=0;t<Agent::T-1;t++)
	{
	
		for(vector<int>::iterator parent_ite=parent_all[list_idx].begin();
								parent_ite!=parent_all[list_idx].end();parent_ite++)
		{
			current_agent = agent_list[*parent_ite];
			int pre_st = current_agent->hidden_state(t);
			int nxt_st = current_agent->hidden_state(t+1);
			sum_theta += current_agent->theta_true[pre_st][nxt_st]/
												current_agent->neighIdx.size();
			sum_theta_mat.col(t) += current_agent->theta_true[pre_st][nxt_st]/
												current_agent->neighIdx.size();
		}
	}
	sum_theta_mat = exp(sum_theta_mat);
}


void Variational::Generate_ROC(vector<Agent*>&agent_list
				,unordered_map<int,int>&obs_dict,vec* allNodes,
				string filename)
{
	Agent* current_agent;
	vec node_list = vec(Agent::N,fill::zeros);
	node_list.load(Agent::nodes_file);
	for(int kk=0;kk<Agent::K;kk++)
	{
		fstream state_file("max_states_"+to_string(kk)
										+"_"
										+filename,fstream::out);
		int agent_idx;
		for(vec::iterator node_label=node_list.begin();node_label!=node_list.end();
				node_label++)
		{
			agent_idx = obs_dict[*node_label];
			current_agent = agent_list[agent_idx];
			state_file<<*node_label<<" ";
			for(int t=0;t<Agent::T;t++)	
			{
				double current_max=-1;
				current_max = current_agent->GetQMarginal(t,kk);
				state_file<<current_max<<" ";
			}
			state_file<<endl;
		}
	
	}
}
