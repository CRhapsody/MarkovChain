import markov_chain
import numpy as np
import math

class SaturatingCounters(markov_chain.MarkovChain):
    def __init__(self, states, transition_dict = None, transition_matrix = None):
        '''
        parameters:
        - states: list of states
        - transition_dict: dictionary of transition probabilities
        - transition_matrix: matrix of transition probabilities, shape = (state_space, state_space, inputs_space)
        '''
        super().__init__(states, transition_dict, transition_matrix)
        if not self.is_prediction_counter():
            raise TypeError("Model is not a member of saturating counters")
    def is_prediction_counter(self):
        for state in self.states:
            for input in self.transition_dict[state]:
                if input == state.outputs:
                    for successor in self.transition_dict[state][input]:
                        if self.states[successor].outputs != state.outputs:
                           return False
                        else:
                            continue
                else:
                    continue
        return True
    # def is_member(self):
    #     '''
    #     Check if the markov chain is a member of the set of saturating counters
    #     '''
    #     # first check if every state receive the same input as its output, the output of its sccessor is the same as its output
        
        
    #     # secondly







class NonPrSaturatingCounters(SaturatingCounters):
    '''
    A Markov chain model of saturating counters, where the state transition is deteministic
    '''
    def __init__(self, states, transition_dict = None, transition_matrix = None):
        '''
        parameters:
        - states: list of states
        - transition_dict: dictionary of transition probabilities
        - transition_matrix: matrix of transition probabilities, shape = (state_space, state_space, inputs_space)
        '''
        super().__init__(states, transition_dict, transition_matrix)
        self.test_transition_deteministic()
    def test_transition_deteministic(self):
        '''
        For the saturating counters, the transition is deterministic, which means the transition matrix has only one 1 in each row, and the rest of the elements are 0.
        '''
        # check if the the number of 1s in each row is 1
        ones_count = np.sum(self.transition_matrix == 1, axis=1)
        if not np.all(ones_count == 1):
            raise TypeError("The transition matrix is not deterministic")
        
        # check if the rest of the elements are 0
        non_ones = self.transition_matrix != 1
        zeros_check = np.all((self.transition_matrix[non_ones] == 0))
        if not zeros_check:
            raise TypeError("The transition matrix is not deterministic")
    
    def is_member(self):
        '''
        Check if this Saturating Counters is a non-probabilistic saturating counters
        which should satisfy the following conditions:
        T (SN, NT) = SN, T (SN, T) = WN1,
        T (ST, T) = ST, T (ST, NT) = WT1,
        T (WNi, T) = WT{i+1}, T (WNi, NT) = WN{i+1},
        T (WTi, T) = WT{i+1}, T (WTi, NT) = WT{i+1}
        '''
        assert self.transition_dict["SN"]["NT"]["SN"] == 1
        assert self.transition_dict["SN"]["T"]["WN1"] == 1
        assert self.transition_dict["ST"]["T"]["ST"] == 1
        assert self.transition_dict["ST"]["NT"]["WT1"] == 1
        for i in range(1, int(math.log2(len(self.states)/2))): # 1,2,...,n-1, n is the bit of the counter
            assert self.transition_dict["WN"+str(i)]["T"]["WT"+str(i+1)] == 1
            assert self.transition_dict["WN"+str(i)]["NT"]["WN"+str(i+1)] == 1
            assert self.transition_dict["WT"+str(i)]["T"]["WT"+str(i+1)] == 1
            assert self.transition_dict["WT"+str(i)]["NT"]["WT"+str(i+1)] == 1



class PrSaturatingCounter(markov_chain.MarkovChain):
    '''
    A Markov chain model of saturating counters, where the state transition is probabilistic
    '''
    def __init__(self, 
                 states, 
                 Probability_threshold,
                 transition_dict = None, transition_matrix = None):
        '''
        :param states: list of states
        :param Probability_threshold: probability threshold, which is m in our paper
        :param transition_dict: dictionary of transition probabilities
        :param transition_matrix: matrix of transition probabilities, shape = (state_space, state_space, inputs_space)
        '''
        super().__init__(states, transition_dict, transition_matrix)
    

    def get_attack_strategy(self, c):
        '''
        Get the attack strategy from the output of attack c and probability threshold m. If c > 2^{n-1}/m, then the probability is higher that the branch of the victim thread is T;
        if c = 2^{n-1}/m, then there is equal probability that the branch of the victim thread is T or NT;
        otherwise it is NT
        '''
        # c = self.run_attack(initial_state, victim_thread_branch)
        n = int(math.log2(len(self.states)/2))
        if c > 2**(n-1)/self.Probability_threshold:
            print("Attacker chooses T")
            return "T"
        elif c == 2**(n-1)/self.Probability_threshold:
            print("Two choices are equally probable") 
            return np.random.choice(["T", "NT"])
        else:
            print("Attacker chooses NT")
            return "NT"
    

class PrivacySaturatingCounter(SaturatingCounters):
    '''
    A Markov chain model of saturating counters, where the state transition is probabilistic
    satisfies differential privacy.
    '''
    def __init__(self, 
                 states, 
                 transition_dict = None, transition_matrix = None):
        '''
        :param states: list of states
        :param transition_dict: dictionary of transition probabilities
        :param transition_matrix: matrix of transition probabilities, shape = (state_space, state_space, inputs_space)
        '''
        super().__init__(states, transition_dict, transition_matrix)
    
    
