from markov_chain import MarkovChain, state, PSCInputs, PSCInputNs, PSCOutputs, PSCOutputNs
import numpy as np
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

if __name__ == "__main__":
    # test_markov_chain_construct_dict()
    # test_markov_chain_construct_matrix()
    test_markov_chain_step()
