from markov_chain import MarkovChain, state, PSCInputs, PSCInputNs, PSCOutputs, PSCOutputNs
import numpy as np
from SaturatingCounters import SaturatingCounters, PrivacySaturatingCounter, NonPrSaturatingCounters, PrSaturatingCounter

class ThreeBitPSC:
    m = 0.1
    states = {
        "SN": state("SN",   PSCOutputs.NT),
        "WN1": state("WN1", PSCOutputs.NT),
        "WN2": state("WN2", PSCOutputs.NT),
        "WN3": state("WN3", PSCOutputs.NT),
        "ST": state("ST",   PSCOutputs.T),
        "WT1": state("WT1", PSCOutputs.T),
        "WT2": state("WT2", PSCOutputs.T),
        "WT3": state("WT3", PSCOutputs.T),        
    },
    transition_dict = {
       "SN": {PSCInputs.NT: {"SN": 1}, 
              PSCInputs.T: {"WN1": m, "SN": 1-m}},
        "WN1": {PSCInputs.NT: {"SN": m, "WN1": 1-m}, 
                PSCInputs.T: {"WN2": m, "WN1": 1-m}},
        "WN2": {PSCInputs.NT: {"WN1": m, "WN2": 1-m}, 
                PSCInputs.T: {"WN3": m, "WN2": 1-m}},
        "WN3": {PSCInputs.NT: {"WN2": m, "WN3": 1-m}, 
                PSCInputs.T: {"ST": m, "WN3": 1-m}},
        "ST": {PSCInputs.T: {"ST": 1}, 
                PSCInputs.NT: {"WT1": m, "ST": 1-m}},
        "WT1": {PSCInputs.T: {"ST": m, "WT1": 1-m}, 
                PSCInputs.NT: {"WT2": m, "WT1": 1-m}},
        "WT2": {PSCInputs.T: {"WT1": m, "WT2": 1-m}, 
                PSCInputs.NT: {"WT3": m, "WT2": 1-m}},
        "WT3": {PSCInputs.T: {"WT2": m, "WT3": 1-m}, 
                PSCInputs.NT: {"SN": m, "WT3": 1-m}}
    }

class PrivacyThreeBitPSC:
    m = 0.1
    p = 0.1
    states = {
        "SN": state("SN",   PSCOutputs.NT),
        "WN1": state("WN1", PSCOutputs.NT),
        "WN2": state("WN2", PSCOutputs.NT),
        "WN3": state("WN3", PSCOutputs.NT),
        "ST": state("ST",   PSCOutputs.T),
        "WT1": state("WT1", PSCOutputs.T),
        "WT2": state("WT2", PSCOutputs.T),
        "WT3": state("WT3", PSCOutputs.T),        
    },
    '''
        P(SN, T, SN) = 1 − m + m*p, P(SN, T, WN1) = m*(1 − p),

        P(SN, NT, SN) = 1 − m + m*(1 − p), P(SN, NT, WN1) = m*p,

        P(WNi, T, WNi+1) = m, P(WNi+1, NT, WNi) = m,

        P(WNi, T, WNi) = 1 − m, P(WNi, NT, WNi) = 1 − m,

        P(ST, T, ST) = 1 − m + m*(1 − p), P(ST, T, WT1) = m*p,

        P(ST, NT, ST) = 1 − m + m*p, P(ST, NT, WT1) = m*(1 − p),

        P(WTi, NT, WTi+1) = m, P(WTi+1, T, WTi) = m,

        P(WTi, T, WTi) = 1 − m, P(WTi, NT, WTi) = 1 − m,
    '''
    transition_dict = {
        "SN":  {PSCInputs.NT: {"SN": 1-m+m*(1-p), "WN1": m*p},
                PSCInputs.T: {"WN1": m*(1-p), "SN": 1-m+m*p}},
        "WN1": {PSCInputs.NT: {"SN": m, "WN1": 1-m}, 
                PSCInputs.T: {"WN2": m, "WN1": 1-m}},
        "WN2": {PSCInputs.NT: {"WN1": 1-m, "WN2": m},
                PSCInputs.T: {"WN3": m, "WN2": 1-m}},
        "WN3": {PSCInputs.NT: {"WN2": 1-m, "WN3": m},
                PSCInputs.T: {"ST": m, "WN3": 1-m}},

        "ST":  {PSCInputs.NT: {"WT1": m*(1-p), "ST": 1-m+m*p},
                PSCInputs.T: {"ST": 1-m+m*(1-p), "WT1": m*p}},
        "WT1": {PSCInputs.T: {"ST": m, "WT1": 1-m}, 
                PSCInputs.NT: {"WT2": m, "WT1": 1-m}},
        "WT2": {PSCInputs.T: {"WT1": m, "WT2": 1-m}, 
                PSCInputs.NT: {"WT3": m, "WT2": 1-m}},
        "WT3": {PSCInputs.T: {"WT2": m, "WT3": 1-m}, 
                PSCInputs.NT: {"SN": m, "WT3": 1-m}}
    }


def test_markov_chain_construct_dict():
    '''
    Test the MarkovChain class
    '''
    # states = [state("A", PSCOutputs.NT), state("B", PSCOutputs.NT), state("C", PSCOutputs.NT)]
    states = [state("A", PSCOutputs.NT), state("B", PSCOutputs.NT), state("C", PSCOutputs.NT)]
    transition_dict = {
        "A": {PSCInputs.NT: {"A": 0.5, "B": 0.5, "C": 0}},
        "B": {PSCInputs.NT: {"A": 0.5, "B": 0.5, "C": 0}},
        "C": {PSCInputs.NT: {"A": 0.5, "B": 0.5, "C": 0}},
    }
    mc = MarkovChain(states, transition_dict)
    print(mc.transition_dict)
    print(mc.transition_matrix)
    print(mc.states)

def test_markov_chain_construct_matrix():
    '''
    Test the MarkovChain class
    '''
    states = [state("A", PSCOutputs.NT), state("B", PSCOutputs.NT), state("C", PSCOutputs.NT)]
    # states = [state("A", PSCInputNs.NT), state("B", PSCInputNs.NT), state("C", PSCInputNs.NT)]
    # transition_matrix = np.array(
    # [[[0.5 0.5 0. ],[0.  0.  0. ]],

    # [[0.5 0.5 0. ],
    # [0.  0.  0. ]],

    # [[0.5 0.5 0. ],
    # [0.  0.  0. ]]])
    transition_matrix = np.array(
    [[[0.5, 0.5, 0.],[0., 0., 0.]],
    [[0.5, 0.5, 0.],[0., 0., 0.]],
    [[0.5, 0.5, 0.],[0., 0., 0.]]])
    mc = MarkovChain(states, transition_matrix = transition_matrix)
    print(mc.transition_dict)
    print(mc.transition_matrix)
    print(mc.states)


def test_markov_chain_step():
    '''
    Test the MarkovChain step
    '''
    m = 0.5
    states = {
        "SN": state("SN",   PSCOutputs.NT),
        "WN1": state("WN1", PSCOutputs.NT),
        "WN2": state("WN2", PSCOutputs.NT),
        "WN3": state("WN3", PSCOutputs.NT),
        "ST": state("ST",   PSCOutputs.T),
        "WT1": state("WT1", PSCOutputs.T),
        "WT2": state("WT2", PSCOutputs.T),
        "WT3": state("WT3", PSCOutputs.T),        
    }
    transition_dict = {
       "SN": {PSCInputs.NT: {"SN": 1}, 
              PSCInputs.T: {"WN1": m, "SN": 1-m}},
        "WN1": {PSCInputs.NT: {"SN": m, "WN1": 1-m}, 
                PSCInputs.T: {"WN2": m, "WN1": 1-m}},
        "WN2": {PSCInputs.NT: {"WN1": m, "WN2": 1-m}, 
                PSCInputs.T: {"WN3": m, "WN2": 1-m}},
        "WN3": {PSCInputs.NT: {"WN2": m, "WN3": 1-m}, 
                PSCInputs.T: {"ST": m, "WN3": 1-m}},
        "ST": {PSCInputs.T: {"ST": 1}, 
                PSCInputs.NT: {"WT1": m, "ST": 1-m}},
        "WT1": {PSCInputs.T: {"ST": m, "WT1": 1-m}, 
                PSCInputs.NT: {"WT2": m, "WT1": 1-m}},
        "WT2": {PSCInputs.T: {"WT1": m, "WT2": 1-m}, 
                PSCInputs.NT: {"WT3": m, "WT2": 1-m}},
        "WT3": {PSCInputs.T: {"WT2": m, "WT3": 1-m}, 
                PSCInputs.NT: {"SN": m, "WT3": 1-m}}
    }
    mc = MarkovChain(states, transition_dict)
    print(mc.transition_dict)
    print(mc.transition_matrix)
    print(mc.states)

    # one step
    # state1step = mc.step(states["SN"], PSCInputs.T)
    # print(state1step)

    # # multiple steps
    # statesteps = mc.run(states["SN"], 
    #                     [PSCInputs.T for _ in range(10)])
    
    # print(mc.run_history)

    # test attack
    count = mc.run_attack(states["SN"], 'T')

def test_saturating_counters():
    '''
    Test the SaturatingCounters class
    '''
    # set m = 1
    # sc = SaturatingCounters(states=ThreeBitPSC.states[0], transition_dict=ThreeBitPSC.transition_dict)
    # nonprpsc = NonPrSaturatingCounters(states=ThreeBitPSC.states[0], transition_dict=ThreeBitPSC.transition_dict)

    # set m = 0.1
    # sc = PrSaturatingCounter(states=ThreeBitPSC.states[0], 
    #                          probability_threshold=0.1,
    #                          transition_dict=ThreeBitPSC.transition_dict)
    # set m,p =0.1
    sc = PrivacySaturatingCounter(states=PrivacyThreeBitPSC.states[0],
                                    probability_threshold=0.1,
                                    privacy_threshold=0.1,
                                    transition_dict=PrivacyThreeBitPSC.transition_dict)

if __name__ == "__main__":
    # test_markov_chain_construct_dict()
    # test_markov_chain_construct_matrix()
    test_markov_chain_step()
#     test_saturating_counters()
