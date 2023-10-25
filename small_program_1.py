import sys
import networkx as nx
import pandas as pd
import numpy as np
import math
import time
import copy
import ntpath
import matplotlib
import matplotlib.pyplot as plt
import pdb
from collections import defaultdict

start_time = time.time()
        

class ReadData(object):

    def __init__(self, file):
        self.file = file
    
    def createTransition_Prob(self):
        print("File in class is ", self.file)
        data = pd.read_csv(self.file)
        print(len(data[:]["s"]))
        u_s = data.s.unique() #unique_states
        u_s = sorted(u_s)
        print("List of Unique States", u_s)
        u_n_s = data.sp.unique() #unique_next_states. Should be identical to unique states but keeping it separate for generalization.
        u_n_s = sorted(u_n_s)
        print("List of Unique Next States", u_n_s)
        u_a = data.a.unique() # unique_actions
        print("List of Unique Actions", u_a)
        
        T_ns_a_s = defaultdict(list)
        N_s_a = defaultdict(list)
        N_s_a_ns = defaultdict(list)
        R_s_a = defaultdict(list)
        rho_s_a = defaultdict(list)
        next_state_from_s = defaultdict(list)
        
        for state in u_s:
            for action in u_a:
                ind_a = str(state) + "_" + str(action)
                #Finding counts of given state and action pair in the data
                df = []
                df = data[(data['s'] == int(state)) & (data['a'] == int(action))]
                N_s_a[ind_a] = len(df)
                #print(df)
                rho_s_a[ind_a] = df[:]['r'].sum()
                R_s_a[ind_a] = rho_s_a[ind_a]/N_s_a[ind_a]
                #print(ind_a, R_s_a[ind_a])
                next_state_from_s[str(state)] = list() 
                #Not all states from s will be accesible therefore next state should only be what is available.
                u_n_s = df.sp.unique()
                for next_state in u_n_s:
                    next_state_from_s[str(state)].append(next_state)
                    ind_b = str(state) + "_" + str(action) + "_" + str(next_state)
                    df_n = df[df['sp'] == int(next_state)]
                    N_s_a_ns[ind_b] = len(df_n)
                    #print(ind_b, N_s_a_ns[ind_b])
                    T_ns_a_s[ind_b] = N_s_a_ns[ind_b]/N_s_a[ind_a]
                    #print(ind_b, T_ns_a_s[ind_b])
                    
        return T_ns_a_s, R_s_a, u_s, u_a, u_n_s, next_state_from_s  
        #T_ns_a_s = np.zeros((len(u_n_s),len(u_a),len(u_s)))
        #N_s_a = 
        
        

def value_iteration(T_ns_a_s, R_s_a, u_s, u_a, u_n_s, next_state_from_s, gamma, epsilon=0.001):
    U1 = {s: 0 for s in u_s}
    #print("U1", U1)
    #print("length U1",len(U1))
    while True:
        U = U1.copy()
        #print("U", U)
        #print("length U",len(U))
        delta = 0
        for s in u_s:
            u_1_temp = 0
            u_for_given_s = list()
            for a in u_a:
                ind_a = str(s) + "_" + str(a)
                sum_t_u = 0
                #print("next_state_from_s", next_state_from_s)
                #print("next_state_from_s[s]", s, next_state_from_s[str(s)])
                for ns in next_state_from_s[str(s)]:
                    ns = int(ns)
                    #print("NEXT STATE", ns, U[ns])
                    ind_b = str(s) + "_" + str(a) + "_" + str(ns)
                    #print("Transition", ind_b, T_ns_a_s[ind_b])
                    if T_ns_a_s[ind_b] == []:
                        T_ns_a_s[ind_b] = 0
                    #print("Transition", ind_b, T_ns_a_s[ind_b])
                    sum_t_u = sum_t_u + float(U[ns])*float(T_ns_a_s[ind_b])
                u_for_given_s.append(R_s_a[ind_a] + gamma*sum_t_u)
            U1[s] = max(u_for_given_s)
            delta = max(delta, abs(U1[s] - U[s]))
            #print("DELTA", delta)
        if delta <= epsilon * (1 - gamma) / gamma:
            #print("Returning U", U)
            return U
        
def best_policy(T_ns_a_s, R_s_a, u_s, u_a, u_n_s, next_state_from_s, U,gamma, filename):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action"""
    pi = {}
    with open(filename, 'w') as f:
        for s in u_s:
            s = str(s)
            pi[s] = max(u_a, key=lambda a: expected_utility(a, s, U, T_ns_a_s, next_state_from_s,R_s_a, gamma))
            #print("Best Policy", s,pi[s])
            f.write("{}\n".format(pi[s]))
    f.close()
    return pi

def expected_utility(a, s, U, T_ns_a_s, next_state_from_s, R_s_a, gamma):
    """Expected utility of executing action a in state s, according to the MDP and U."""
    sum_t_u = 0 
    ind_a = str(s) + "_" + str(a)
    for ns in next_state_from_s[str(s)]:
                    ns = int(ns)
                    #print("NEXT STATE", ns, U[ns])
                    ind_b = str(s) + "_" + str(a) + "_" + str(ns)
                    #print("Transition", ind_b, T_ns_a_s[ind_b])
                    if T_ns_a_s[ind_b] == []:
                        T_ns_a_s[ind_b] = 0
                    #print("Transition", ind_b, T_ns_a_s[ind_b])
                    sum_t_u = sum_t_u + float(U[ns])*float(T_ns_a_s[ind_b])
                    #print("SUM", sum_t_u)
    u = R_s_a[ind_a] + gamma*sum_t_u
    #print("STATE, ACTION, UTILITY",s, a, u)
    return u
                    
    #return sum(p * U[s1] for (p, s1) in T_ns_a_s)

def main():
    inputfilename = "C:/Users/mink_/AA228Student/workspace/project2/small.csv"
    outfilename = "C:/Users/mink_/AA228Student/workspace/project2/small.policy"
    print("INPUT ", inputfilename)
    data = ReadData(inputfilename)
    T_ns_a_s, R_s_a, u_s, u_a, u_n_s, next_state_from_s = data.createTransition_Prob()
    gamma = 0.95
    train_time_start = time.time()
    U = value_iteration(T_ns_a_s, R_s_a, u_s, u_a, u_n_s, next_state_from_s, gamma)
    train_time_end = time.time()
    print("Train time is %s seconds ---" % (train_time_end - train_time_start))
    best_policy(T_ns_a_s, R_s_a, u_s, u_a, u_n_s, next_state_from_s, U, gamma,outfilename)
    print("DONEE")
    print("Time taken to get the output %s seconds ---" % (time.time() - start_time))
    
if __name__ == '__main__':
    main()

