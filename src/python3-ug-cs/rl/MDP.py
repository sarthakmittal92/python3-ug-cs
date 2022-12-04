import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, value, LpStatusOptimal
from sys import maxsize

# state
class State:
    
    # initialise
    def __init__(self, i):
        self.i = i # state number
        self.A = set() # actions that can be taken

# MDP
class myMDP:
    
    # initialise
    def __init__(self, S, A, e):
        self.ns = S # number of states
        self.na = A # number of actions
        self.S = {i:State(i) for i in range(S)} # all states
        self.A = [i for i in range(A)] # all actions
        self.e = e # end states
        self.T = [[[]for _ in range(A)] for _ in range(S)] # transitions
        self.thres = 1e-10 # threshold to stop
    
    # set parameters
    def setParams(self, g, t):
        self.g = g # gamma
        self.t = t # type
    
    # policy evaluation
    def policyEvaluation(self, policy):
        modulo = self.ns
        if len(policy) != self.ns:
            modulo = len(policy)
        V = [0 for _ in range(self.ns)] # value function
        curr = None # copy of value function
        while True:
            curr = V.copy()
            for s1 in self.S: # iterate over states and find V
                if s1 in self.e: # end state
                    V[s1] = 0
                    continue
                Vs1 = 0
                trans = self.T[s1][policy[s1 % modulo]] # get transition data
                for (s2,r,p) in trans: # compute value
                    if s2 not in self.e:
                        Vs1 += p * (r + self.g * curr[s2])
                    else:
                        Vs1 += p * r
                V[s1] = Vs1
            if np.linalg.norm(np.array(curr) - np.array(V)) <= self.thres: # check threshold
                return V, policy
    
    # value iteration
    def valueIteration(self):
        V = [0 for _ in range(self.ns)] # value function
        curr = None # copy of value function
        policy = [0 for _ in range(self.ns)] # policy
        while True:
            curr = V.copy()
            for s1 in self.S: # iterate over states and find V
                if s1 in self.e: # end state
                    V[s1] = 0
                    continue
                optimal = 0
                maxV = -maxsize
                for a in sorted(self.S[s1].A): # iterate over actions
                    trans = self.T[s1][a] # get transition data
                    Vs1 = 0
                    for (s2,r,p) in trans: # compute value
                        if s2 not in self.e:
                            Vs1 += p * (r + self.g * curr[s2])
                        else:
                            Vs1 += p * r
                    if Vs1 > maxV:
                        maxV = Vs1
                        optimal = a
                if len(self.S[s1].A) == 0:
                    maxV = 0
                V[s1] = maxV # set value
                policy[s1] = optimal # set policy
            if np.linalg.norm(np.array(curr) - np.array(V)) <= self.thres: # check threshold
                for s in self.S: # iterate over states and check/assign policy
                    if len(self.S[s].A) == 0 or (policy[s] in self.S[s].A):
                        continue
                    for a in self.S[s].A:
                        policy[s] = a
                        break
                return V, policy
    
    # Howard's policy iteration
    def howardsPolicyIteration(self):
        policy = [] # policy
        for s in self.S: # iterate over states and get initial policy
            if len(self.S[s].A) == 0:
                policy.append(0)
            else:
                for a in sorted(self.S[s].A):
                    policy.append(a)
                    break
        i = 0
        while i < 10: # 10 improvement iterations
            i += 1
            V, policy = self.policyEvaluation(policy)
            policy1 = policy.copy() # check policy
            for s1 in self.S: # iterate over states and find Q
                if s1 in self.e: # end state
                    continue
                currQ = V[s1]
                maxQ = currQ
                optimal = policy[s1]
                for a in sorted(self.S[s1].A): # iterate over actions
                    Qs1a = 0
                    trans = self.T[s1][a] # get transition data
                    for (s2,r,p) in trans: # compute value
                        if s2 not in self.e:
                            Qs1a += p * (r + self.g * V[s2])
                        else:
                            Qs1a += p * r
                    if abs(Qs1a - currQ) >= self.thres and Qs1a > maxQ:
                        optimal = a
                        maxQ = Qs1a
                policy1[s1] = optimal # set optimal
            if np.array_equal(policy,policy1): # reached optimal
                for s in self.S: # iterate over states and check/assign policy
                    if len(self.S[s].A) == 0 or (policy[s] in self.S[s].A):
                        continue
                    for a in self.S[s].A:
                        policy[s] = a
                        break
                return V, policy
            policy = policy1 # use improved policy
        return V, policy
    
    # linear programming
    def linearProgramming(self):
        prob = LpProblem('V',LpMinimize) # problem
        V = [LpVariable('V' + str(i)) for i in range(self.ns)] # value function
        prob += lpSum(V) # value function constraint
        for s1 in self.S: # iterate over states and add constraints
            for a in self.S[s1].A: # iterate over actions
                trans = self.T[s1][a] # get transition data
                Qsa = []
                for (s2,r,p) in trans: # compute value
                    if s2 not in self.e:
                        Qsa.append(p * (r + self.g * V[s2]))
                    else:
                        Qsa.append(p * r)
                if len(trans) != 0:
                    prob += V[s1] >= lpSum(Qsa) # constraint for optimal
            if len(self.S[s1].A) == 0:
                prob += V[s1] >= 0 # non-negative constraint
        res = prob.solve(PULP_CBC_CMD(msg = 0)) # solve LP
        assert res == LpStatusOptimal # assert optimality
        Vf = [value(x) for x in V]
        policy = [0 for _ in range(self.ns)]
        for s1 in self.S: # iterate over states to assign policy
            Qs = [-maxsize for _ in range(self.na)]
            for a in self.S[s1].A: # iterate over actions
                trans = self.T[s1][a] # get transition data
                Qs[a] = 0
                for (s2,r,p) in trans: # compute value
                    if s2 not in self.e:
                        Qs[a] += p * (r + self.g * Vf[s2])
                    else:
                        Qs[a] += p * r
            policy[s1] = np.argmax(Qs) # set policy
        for s in self.S: # iterate over states and check/assign policy
            if len(self.S[s].A) == 0 or (policy[s] in self.S[s].A):
                continue
            for a in self.S[s].A:
                policy[s] = a
                break
        return Vf, policy
    
    def solver(self, policy, algo):
        if len(policy) != 0: # policy file given
            return self.policyEvaluation(policy)
        else:
            if algo == 'vi': # value iteration
                return self.valueIteration()
            elif algo == 'hpi': # Howard's policy iteration
                return self.howardsPolicyIteration()
            elif algo == 'lp': # linear programming
                return self.linearProgramming()