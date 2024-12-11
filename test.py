from markov_chain import MarkovChain, state, PSCInputs, PSCInputNs, PSCOutputs, PSCOutputNs
import numpy as np
def test_markov_chain_construct_dict():
    '''
    Test the MarkovChain class
    '''
    # states = [state("A", PSCOutputs.NT), state("B", PSCOutputs.NT), state("C", PSCOutputs.NT)]
    states = [state("A", PSCInputs.NT), state("B", PSCInputs.NT), state("C", PSCInputs.NT)]
    transition_dict = {
        "A": {PSCOutputs.NT: {"A": 0.5, "B": 0.5, "C": 0}},
        "B": {PSCOutputs.NT: {"A": 0.5, "B": 0.5, "C": 0}},
        "C": {PSCOutputs.NT: {"A": 0.5, "B": 0.5, "C": 0}},
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


# def test_markov_chain_step

if __name__ == "__main__":
    # test_markov_chain_construct_dict()
    test_markov_chain_construct_matrix()