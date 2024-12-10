import numpy as np
from enum import Enum

class PSCInputs(Enum):
    NT = "NT"
    T = "T"



class state:
    def __init__(self, name, output):
        self.name = name
        # self.inputs = inputs
        self.__outputs = output
    @property
    def outputs(self):
        return self.__outputs
    @outputs.setter
    def outputs(self, value):
        self.__outputs = value
    def __repr__(self):
        return f"state({self.name}, {self.outputs})"
    def __str__(self):
        return f"{self.name}"
    def __eq__(self, other):
        return self.name == other.name
    def __hash__(self):
        return hash(self.name)
    def __len__(self):
        return len(self.name)
    # def __getitem__(self, key):
    #     return self.name[key]






class MarkovChain:
    def __init__(self, states, transition_dict = None):
        '''
        parameters:
        - states: list of states
        - transition_dict: dictionary of transition probabilities
        '''
        self.transition_dict = transition_dict
        self.states = states
        # self.Inputs = Inputs
        self.state_space = len(states)
        self.construct_transition_map(transition_dict)
    

    def construct_transition_map(self, transition_dict):
        '''
        parameters:
        - transition_dict: dictionary of transition probabilities
        '''
        if transition_dict is None:
            print ("No transition dictionary is provided, construct empty transition dictionary")
            self.transition_dict = {}
            for state in self.states:
                self.transition_dict[state] = {}
                for input in PSCInputs:
                    self.transition_dict[state][input] = {}
        else:
            self.transition_dict = transition_dict

    def set_transition_for_state(self, from_state, transition_dict):
        '''
        parameters:
        - form_state: state str that transition probabilities are set for
        - transition_dict: dictionary of transition probabilities,such as  {"input1": {"to state11": probability11, "to state12": probability12}, "input2": {"to state21": probability21, "to state22": probability22}......}
        '''

        if from_state not in self.transition_dict: 
            raise ValueError(f"Invalid from state {from_state}")
        for input, next_states in transition_dict.items():
            if isinstance(input, PSCInputs):
                raise ValueError(f"Invalid input {input}")
            for next_state, probability in next_states.items():
                if next_state not in self.states:
                    raise ValueError(f"Invalid next state {next_state}")
                self.transition_dict[from_state][input][next_state] = probability
    def step(self, from_state, input):
        states = self.transition_dict[from_state][input].keys()
        probabilities = self.transition_dict[from_state][input].values()
        next_state = np.random.choice(states, 1, p=probabilities)


        return next_state[0]

        # self.state = np.random.choice(self.states, p=self.transition_matrix[self.state])
        # self.state_history.append(self.state)
        
    def run(self, initial_state, steps, input_sequence):
        '''
        :param initial_state: initial state of the Markov chain
        :param steps: number of steps to run the Markov chain
        :param input_sequence: list of inputs to apply at each step
        '''
        self.run_history = [initial_state]
        state = initial_state
        for i in range(steps):
            state = self.step(state, input_sequence[i])
            self.run_history.append(state)
            
    def __getitem__(self, key):
        return self.states[key]
    

if __name__ == "__main__":
    m = 0.5
    # state = ["SN", "WN1", "WN2", "WN3", 
    #          "ST", "WT1", "WT2", "WT3"]
    # state_list = [state("SN", PSCInputs.NT),
    #             state("WN1", PSCInputs.NT),
    #             state("WN2", PSCInputs.NT),
    #             state("WN3", PSCInputs.NT),
    #             state("ST",  PSCInputs.T),
    #             state("WT1", PSCInputs.T),
    #             state("WT2", PSCInputs.T),
    #             state("WT3", PSCInputs.T)]
    state_dict = {
        "SN": state("SN", PSCInputs.NT),
        "WN1": state("WN1", PSCInputs.NT),
        "WN2": state("WN2", PSCInputs.NT),
        "WN3": state("WN3", PSCInputs.NT),
        "ST": state("ST", PSCInputs.T),
        "WT1": state("WT1", PSCInputs.T),
        "WT2": state("WT2", PSCInputs.T),
        "WT3": state("WT3", PSCInputs.T),        
    }
    # transition matrix like this: P(SN, NT, SN) = 1
    # one example 3bit PSC model
    transition_dict = {
       "SN": {PSCInputs.NT: {"SN": 1}, 
              PSCInputs.T: {"WN1": m, "SN": 1-m}},
        "WN1": {PSCInputs.NT: {"SN": m, "WN1": 1-m}, 
                PSCInputs.T: {"WN2": m, "WN1": 1-m}},
        "WN2": {PSCInputs.NT: {"WN1": m, "WN2": 1-m}, 
                PSCInputs.T: {"WN3": m, "WN2": 1-m}},
        "WN3": {PSCInputs.NT: {"WN2": m, "WN3": 1-m}, 
                PSCInputs.T: {"ST": m, "WN3": 1-m}},
        "ST": {PSCInputs.NT: {"WN3": m, "ST": 1-m}, 
                PSCInputs.T: {"WT1": m, "ST": 1-m}},
        "WT1": {PSCInputs.NT: {"ST": m, "WT1": 1-m}, 
                PSCInputs.T: {"WT2": m, "WT1": 1-m}},
        "WT2": {PSCInputs.NT: {"WT1": m, "WT2": 1-m}, 
                PSCInputs.T: {"WT3": m, "WT2": 1-m}},
        "WT3": {PSCInputs.NT: {"WT2": m, "WT3": 1-m}, 
                PSCInputs.T: {"SN": m, "WT3": 1-m}}
    }
    model = MarkovChain(states = state_dict.values(), transition_dict = transition_dict)
    # test set_transition_for_state funciton
    model.set_transition_for_state(state_dict["SN"], {PSCInputs.NT: {state_dict["SN"]: 1, state_dict["WN1"]: m}, PSCInputs.T: {state_dict["SN"]: 1-m, state_dict["WN1"]: m}})



       