import numpy as np
import math
from multiprocessing import Pool

# Bernoulli Arm
class BernoulliArm:
    
    # initialise
    def __init__(self, p):
        self.p = p
    
    # pull
    def pull(self, num_pulls = None):
        return np.random.binomial(1, self.p, num_pulls)

# Bernoulli bandit
class BernoulliBandit:
    
    # initialise
    def __init__(self, probs = [0.3, 0.5, 0.7], batch_size = 1):
        self.__arms = [BernoulliArm(p) for p in probs]
        self.__batch_size = batch_size
        self.__max_p = max(probs)
        self.__regret = 0
    
    # pull
    def pull(self, index):
        assert self.__batch_size == 1, "'pull' can't be called for in batched setting, use 'batch_pull' instead"
        reward = self.__arms[index].pull()
        self.__regret += self.__max_p - reward
        return reward

    # batched pull
    def batch_pull(self, indices, num_pulls):
        assert sum(num_pulls) == self.__batch_size, f"total number of pulls should match batch_size of {self.__batch_size}"
        rewards = {}
        for i, np in zip(indices, num_pulls):
            rewards[i] = self.__arms[i].pull(np)
            self.__regret += (self.__max_p * np - rewards[i].sum())
        return rewards
    
    # regret
    def regret(self):
        return self.__regret
    
    # batch size
    def batch_size(self):
        return self.__batch_size
    
    # number of arms
    def num_arms(self):
        return len(self.__arms)

# base class
class Algorithm:
    
    # initialise
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    # pull
    def give_pull(self):
        raise NotImplementedError
    
    # reward
    def get_reward(self, arm_index, reward):
        raise NotImplementedError
    
    # function to calculate empirical means using rewards and pulls
    def getEmpMean(self, rewards, pulls):
        empMean = []
        for i in range(self.num_arms):
            if pulls[i] == 0:
                # arm has not yet been pulled
                empMean.append(1e5)
            else:
                # updated empirical mean
                empMean.append(rewards[i] / pulls[i])
        # convert to numpy array
        return np.array(empMean)
    
    # function to calculate extra term in UCB
    def getUCBUncert(self, time, pulls):
        # numerator to find horizon/time factor
        num = math.sqrt(2 * math.log(time))
        uncert = []
        for i in range(self.num_arms):
            if pulls[i] == 0:
                # arm has not yet been pulled
                uncert.append(1e5)
            else:
                # updated uncertainty
                uncert.append(num / math.sqrt(pulls[i]))
        # convert to numpy array
        return np.array(uncert)
    
    # function to calculate KL-divergence
    def KL(self, p, q):
        # base case 1
        if p == 0:
            return math.log(1 / (1 - q))
        # base case 2
        if p == 1:
            return math.log(1 / q)
        # full expression
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    
    # function to binary search for maximum value
    def getUCBKLUncert(self, time, c, p, pulls):
        if (pulls == 0):
            # arm has not yet been pulled
            return (1 + p) / 2
        # upper bound for the divergence
        bound = (math.log(time) + c * math.log(math.log(time))) / pulls
        l = p
        r = 1
        # searching for the largest allowed value
        while r - l > 1e-3:
            q = (l + r) / 2
            # find divergence
            kl = self.KL(p,q)
            if kl < bound:
                # within bound so move interval ahead
                l = q
            else:
                # out of bound so move interval behind
                r = q
        # found value
        return (l + r) / 2

# epsilon-greedy
class Eps_Greedy(Algorithm):
    
    # initialise
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    # pull
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    # reward
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value     

# ucb
class UCB(Algorithm):
    
    # initialise
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.num_pulls = 0
        self.pulls = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
    
    # pull
    def give_pull(self):
        self.num_pulls += 1
        # get the UCB value for this arm at the given time (modeled by number of pulls)
        ucb = self.getEmpMean(self.rewards,self.pulls) + self.getUCBUncert(self.num_pulls,self.pulls)
        # return index of the largest/optimal
        return np.argmax(ucb)
    
    # reward
    def get_reward(self, arm_index, reward):
        # update the pulls and record reward obtained
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward

# kl-ucb
class KL_UCB(Algorithm):
    
    # initialise
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.num_pulls = 0
        self.pulls = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        self.c = 3
    
    # pull
    def give_pull(self):
        self.num_pulls += 1
        # get the empirical means
        empMean = self.getEmpMean(self.rewards,self.pulls)
        ucbkl = []
        for i in range(self.num_arms):
            # iterate over the arms and find the maximum value for ucb-kl for each
            ucbkl.append(self.getUCBKLUncert(self.num_pulls,self.c,empMean[i],self.pulls[i]))
        # return index of the largest/optimal
        return np.argmax(np.array(ucbkl))
    
    # reward
    def get_reward(self, arm_index, reward):
        # update the pulls and record reward obtained
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward

# thompson
class Thompson_Sampling(Algorithm):
    
    # initialise
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.pulls = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
    
    # pull
    def give_pull(self):
        # get the values using beta distribution on each
        thmpsn = np.random.beta(self.rewards + 1,self.pulls - self.rewards + 1)
        # return index of the largest/optimal
        return np.argmax(thmpsn)
    
    # reward
    def get_reward(self, arm_index, reward):
        # update the pulls and record reward obtained
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward

# batched
class AlgorithmBatched:
    
    # initialise
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"
        self.pulls = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
    
    # pull
    def give_pull(self):
        choices = {}
        for _ in range(self.batch_size):
            # choose random arm for each iteration of batch size
            thmpsn = np.random.beta(self.rewards + 1,self.pulls - self.rewards + 1)
            arm = np.argmax(thmpsn)
            # record the choice into a dictionary
            choices.setdefault(arm,0)
            choices[arm] += 1
        # return lists of indices and counts
        return np.array(list(choices.keys())), np.array(list(choices.values()))
    
    # reward
    def get_reward(self, arm_rewards):
        # update the pulls and record reward obtained
        for arm, res in arm_rewards.items():
            self.pulls[arm] += len(res)
            self.rewards[arm] += np.sum(res)

# many arms
class AlgorithmManyArms:
    
    # initialise
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        # Horizon is same as number of arms
        # You can add any other variables you need here
        self.pulls = np.zeros(self.num_arms)
        self.rewards = np.zeros(self.num_arms)
        # choosing factor to accept probability to some extent beyond maximum
        self.exploit = 0.92 + np.random.random() / 20
        self.thres = ((self.num_arms - 1) / self.num_arms) * self.exploit
        # recording the means and the current choice
        self.means = np.ones(self.num_arms) * (self.num_arms - 1) / 2
        self.optimal = np.random.randint(self.num_arms)
    
    # pull
    def give_pull(self):
        # check if the current choice is within threshold
        if self.means[self.optimal] >= self.thres:
            return self.optimal
        # choose a new arm randomly
        self.optimal = np.random.randint(self.num_arms)
        return self.optimal
    
    # reward
    def get_reward(self, arm_index, reward):
        # update the pulls and means and record reward obtained
        self.pulls[arm_index] += 1
        self.rewards[arm_index] += reward
        self.means[arm_index] = self.rewards[arm_index] / self.pulls[arm_index]

def single_sim(seed=0, ALGO=Algorithm, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
    np.random.seed(seed)
    np.random.shuffle(PROBS)
    bandit = BernoulliBandit(probs=PROBS)
    algo_inst = ALGO(num_arms=len(PROBS), horizon=HORIZON)
    for t in range(HORIZON):
        arm_to_be_pulled = algo_inst.give_pull()
        reward = bandit.pull(arm_to_be_pulled)
        algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
    return bandit.regret()

def single_batch_sim(seed=0, ALGO=Algorithm, PROBS=[0.3, 0.5, 0.7], HORIZON=1000, BATCH_SIZE=1):
    np.random.seed(seed)
    np.random.shuffle(PROBS)
    bandit = BernoulliBandit(probs=PROBS, batch_size=BATCH_SIZE)
    algo_inst = ALGO(num_arms=len(PROBS),horizon=HORIZON, batch_size=BATCH_SIZE)
    for _ in range(HORIZON//BATCH_SIZE):
        indices, num_pulls = algo_inst.give_pull()
        rewards_dict = bandit.batch_pull(indices, num_pulls)
        algo_inst.get_reward(rewards_dict)
    return bandit.regret()

def simulate(algorithm, probs, horizon, num_sims=50):
    def multiple_sims(num_sims=50):
        with Pool(10) as pool:
            regrets = pool.starmap(single_sim,
                [(i, algorithm, probs, horizon) for i in range(num_sims)])
        return regrets
    return np.mean(multiple_sims(num_sims))

def batch_simulate(algorithm, probs, horizon, batch_size, num_sims=50):
    def multiple_sims(num_sims=50):
        with Pool(10) as pool:
            regrets = pool.starmap(single_batch_sim,
                [(i, algorithm, probs, horizon, batch_size) for i in range(num_sims)])
        return regrets
    return np.mean(multiple_sims(num_sims))