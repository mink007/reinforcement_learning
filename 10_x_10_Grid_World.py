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
        
def Q_learning(discount_factor, alpha, inputfilename):
    data = pd.read_csv(inputfilename)
    #Initialize Q_s_a Dictionary
    Q_s_a = defaultdict(list)
    #Get list of unique states from data
    u_s = data.s.unique() 
    #Get list of unique actions available from data
    u_a = data.a.unique()
    #pdb.set_trace()

    #Run the iteration n number of times. Picked 4
    for loop in range(1,70):
        #Iterate over each row in the data
         for t in range(0,len(data[:]["s"])):
             #Initialize the state to the state in the row 't' in the data.
             state = data['s'][t]
             #Get action from row 't'
             action = data['a'][t]
             #Get Reward from row 't'
             R = data['r'][t]
             #Get next state from row 't'
             next_state = data['sp'][t]
             #state and action Index for the Q_s_a dictionary
             index_s_a = str(state) + "_" + str(action)
             
             #Get maximum Q_s_a for the next_state from all available actions
             #Note: Q_s_a will be 0 in iterations where this was not computed.
             # Will run the for 't' loop some iterations where Q_s_a will appear initialized.
             
             #Get the action for next state that gives maximum Q_s_a
             max_Q_ns_action = max(u_a, key=lambda a: return_Q_ns_a(next_state, a, Q_s_a))
             #print("ACTION IS", max_Q_ns_action)
             
             
             #Create next_state and action index that gives maximum Q_s_a
             index_max_Q_ns_a = str(next_state) + "_" + str(max_Q_ns_action)
             
             #Find maximum Q_s_a value using the next_state and action
             if str(index_max_Q_ns_a) not in Q_s_a.keys():
                 #print("Initializing Q_s_a for next state and action to 0 as key not found yet")
                 Q_s_a[index_max_Q_ns_a] = 0
                 max_Q_ns_a = 0
             else:
                 #print("index_max_Q_ns_a Key present")
                 max_Q_ns_a = Q_s_a[index_max_Q_ns_a]
             #print("max_Q_ns_a is", max_Q_ns_a)
             
             if str(index_s_a) not in Q_s_a.keys():
                 #print("Initializing Q_s_a for state and action to 0 as key not found yet")
                 Q_s_a[index_s_a] = 0
                 #Track a counter to ensure all keys are present and initialized
                 
             #Find factorized reward to be added to Q_s_a for given state and action.
             factorized_reward = alpha*(R + discount_factor*max_Q_ns_a - Q_s_a[index_s_a])
             #Update Q_s_a for given state and action pair
             Q_s_a[index_s_a] = Q_s_a[index_s_a] + factorized_reward
    
    #pdb.set_trace()
    #print("returning Q_s_a", Q_s_a)
    return Q_s_a, u_s, u_a
                       
def return_Q_ns_a(next_state, a, Q_s_a):
    ind_a = str(next_state) + "_" + str(a)
    if str(ind_a) not in Q_s_a.keys():
        #print("Returning 0 as key not found in dictionary yet")
        return 0
    #print("Returning Q_s_a", ind_a, Q_s_a[ind_a])
    return Q_s_a[ind_a]
    
               
def best_policy(Q_s_a, u_s, u_a, filename,num_states):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action"""
    pi = {}
    
    u_s=sorted(u_s)
    #pdb.set_trace()
    #Fill up missing states information and assign available state information to them in a loop.
    new_u_s = defaultdict(list)
    k = 1
    while k <= num_states:
        for s in u_s:
            new_u_s[k] = s
            k = k + 1
            if k > num_states:
                #print("Breaking")
                break
                break
                
                
    #print("While loop exited")
    print("LENGTH", len(new_u_s))
    #pdb.set_trace()
    i = 1
    with open(filename, 'w') as f:
        for st in range(1,len(new_u_s)+1,1): 
            #print("CURRENT STATE, i", new_u_s[st], i)
            if st in u_s:
                #print("Writing", st)
                #Use the state if it is in the data
                s = str(st)
            else:
                s = str(new_u_s[st])
                #pi[s] = s
                #print("Writing to file", pi[s])
                #f.write("{}\n".format(pi[s]))
                #continue
            #print("SSSS", s)
            pi[s] = max(u_a, key=lambda a: return_Q_ns_a(s, a, Q_s_a))
            #print("Best Policy", s,pi[s])
            f.write("{}\n".format(pi[s]))
            i = i + 1
    f.close()
    return pi

def main():
    inputfilename = "C:/Users/mink_/AA228Student/workspace/project2/small.csv"
    outfilename = "C:/Users/mink_/AA228Student/workspace/project2/small.policy"
    print("INPUT ", inputfilename)
    gamma = 0.95
    alpha = 0.01
    train_time_start = time.time()
    Q_s_a, u_s, u_a = Q_learning(gamma, alpha, inputfilename)
    train_time_end = time.time()
    print("Train time is %s seconds ---" % (train_time_end - train_time_start))
    num_states = 100
    best_policy(Q_s_a, u_s, u_a, outfilename,num_states)
    print("DONEE")
    print("Time taken to get the output %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()

