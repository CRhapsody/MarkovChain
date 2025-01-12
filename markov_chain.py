import numpy as np
from enum import Enum
import time

class PSCInputs(Enum):
    NT = "NT"
    T = "T"

    def __str__(self):
        return self.value
    def __repr__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, PSCInputs):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        elif isinstance(other, PSCOutputs):
            return self.value == other.value
    def __hash__(self):
        return hash(self.value)
        

class PSCInputNs(Enum):
    NT = 0
    T = 1

    def __repr__(self) -> str:
        if self == PSCInputNs.NT:
            return "NT"
        elif self == PSCInputNs.T:
            return "T"
    def __str__(self):
        if self == PSCInputNs.NT:
            return "NT"
        elif self == PSCInputNs.T:
            return "T"

class PSCOutputs(Enum):
    NT = "NT"
    T = "T"

    def __str__(self):
        return self.value
    def __repr__(self):
        return self.value
    def __eq__(self, other):
        if isinstance(other, PSCOutputs):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        elif isinstance(other, PSCInputs):
            return self.value == other.value
    def __hash__(self):
        return hash(self.value)

class PSCOutputNs(Enum):
    NT = 0
    T = 1

    def __repr__(self) -> str:
        if self == PSCOutputNs.NT:
            return "NT"
        elif self == PSCOutputNs.T:
            return "T"
    def __str__(self):
        if self == PSCOutputNs.NT:
            return "NT"
        elif self == PSCOutputNs.T:
            return "T"


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
        # return f"state({self.name}, {self.outputs})"
        return self.name
    def __str__(self):
        return f"{self.name}"
    def __eq__(self, other):
        if isinstance(other, state):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
    def __hash__(self):
        return hash(self.name)
    def __len__(self):
        return len(self.name)
    # def __getitem__(self, key):
    #     return self.name[key]






class MarkovChain:
    def __init__(self, states, transition_dict = None, transition_matrix = None):
        '''
        parameters:
        - states: list of states or dictionary of states, list: [state1, state2, ...], dictionary: {state1.name: state1, state2.name: state2, ...}
        - transition_dict: dictionary of transition probabilities
        - transition_matrix: numpy matrix of transition probabilities, shape = (state_space, inputs_space, state_space)
        '''
        # self.transition_dict = transition_dict
        self.states = states
        self.map_state_into_index()
        # self.Inputs = Inputs
        self.state_space = len(states)
        self.construct_transition_dict(transition_dict)
        self.construct_transition_matrix(transition_matrix)
        if (transition_dict is None) and (transition_matrix is None):
            raise ValueError("No transition dictionary or matrix is provided")
        elif transition_dict is not None:
            self.transfer_transition_dict_to_matrix()
        elif transition_matrix is not None:
            self.transfer_transition_matrix_to_dict()
    
    def map_state_into_index(self):
        if isinstance(self.states, list):
            if isinstance(self.states[0], state):
                self.state_index = {state.name: i for i, state in enumerate(self.states)}
            elif isinstance(self.states[0], str):
                self.state_index = {state: i for i, state in enumerate(self.states)}
        elif isinstance(self.states, dict):
            self.state_index = {state.name: i for i, state in enumerate(self.states.values())}



    def construct_transition_dict(self, transition_dict):
        '''
        parameters:
        - transition_dict: dictionary of transition probabilities, such as {"state1": {"input1": {"to state11": probability11, "to state12": probability12}, "input2": {"to state21": probability21, "to state22": probability22}......}, "state2": {"input1": {"to state11": probability11, "to state12": probability12}, "input2": {"to state21": probability21, "to state22": probability22}......}, ...}
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
    
    def construct_transition_matrix(self, transition_matrix):
        '''
        parameters:
        - transition_matrix: matrix of transition probabilities, shape = (state_space, state_space, inputs_space)
        '''
        if transition_matrix is None:
            print ("No transition matrix is provided, construct empty transition matrix")
            self.transition_matrix = np.zeros((self.state_space, len(PSCInputs), self.state_space))
        else:
            self.transition_matrix = transition_matrix

    def transfer_transition_dict_to_matrix(self):
        '''
        transfer transition dictionary to transition matrix
        '''
        for from_state, inputs in self.transition_dict.items():
            for input, next_states in inputs.items():
                # convert input to PSCInputNs
                if input == PSCInputs.NT:
                    input = PSCInputNs.NT
                elif input == PSCInputs.T:
                    input = PSCInputNs.T
                    
                for next_state, probability in next_states.items():
                    self.transition_matrix[self.state_index[from_state], input.value, self.state_index[next_state]] = probability
    
    def transfer_transition_matrix_to_dict(self):
        '''
        transfer transition matrix to transition dictionary
        '''
        for from_state_index, input_index, to_state_index in np.ndindex(self.transition_matrix.shape):
            self.transition_dict[self.states[from_state_index]][PSCInputs(PSCInputNs(input_index).name)][self.states[to_state_index]] = self.transition_matrix[from_state_index, input_index, to_state_index]       


    def set_transition_for_state_for_dict(self, from_state, transition_dict):
        '''
        set transition probabilities for a state for a dictionary

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
                # need to update transition matrix
                self.transition_matrix[self.state_index[from_state], PSCInputNs[input], self.state_index[next_state]]  = probability
    
    def set_transition_for_state_for_matrix(self, transition_matrix):
        '''

        set transition probabilities for a state for a matrix

        parameters:
        - form_state: state str that transition probabilities are set for
        - transition_matrix: matrix of transition probabilities, shape = [(from_state1_index, to_state1_index, input1, probability1), (from_state2_index, to_state2_index, input2, probability2), ...]
        '''
        for from_state_index, to_state_index, input_index, probability in transition_matrix:
            self.transition_matrix[from_state_index, input_index, to_state_index] = probability
            # need to update transition dictionary
            self.set_transition_for_state_for_dict(self.states[from_state_index], {PSCInputs(input_index): {self.states[to_state_index]: probability}})


    def step(self, from_state, input):
        # if isinstance(from_state, str):
        #     from_state = self.states[self.state_index[from_state]]
        states = list(self.transition_dict[from_state][input].keys())
        probabilities = list(self.transition_dict[from_state][input].values())
        next_state = np.random.choice(states, 1, p=probabilities)


        return next_state[0]

        # self.state = np.random.choice(self.states, p=self.transition_matrix[self.state])
        # self.state_history.append(self.state)
        
    def run(self, initial_state, input_sequence):
        '''
        :param initial_state: initial state of the Markov chain
        :param steps: number of steps to run the Markov chain
        :param input_sequence: list of inputs to apply at each step
        '''
        self.run_history = [initial_state]
        state = initial_state
        for i in range(len(input_sequence)):
            state = self.step(state, input_sequence[i])
            self.run_history.append(state)
        
        return state
    
    def run_attack(self, 
                initial_state, 
                victim_thread,
                input_sequence=None,
                is_probe_step_number = None
                ):
        '''
        execute Prime+Probe Attack 

        :param initial_state: state to start the attack
        :param victim_thread: the victim thread, which is T or NT
        :param input_sequence: list of inputs to apply at each step. If None, use the default input sequence
        :param is_probe_step_number: the number of steps to probe the target. If None, count the number of steps to probe the target

        :return count: the number of steps to probe the target
        :return len(self.run_history): if input_sequence is not None, return the length of run_history
        '''
        # step 1: Prime the target
        # The attacker first primes the target saturating counters to an initial state (e.g., ST) 
        # with successive taken branches and then waits for the victim to execute.
        i = 1
        now_state = initial_state
        while True:

            now_state = self.step(now_state, PSCInputs.T)
            if now_state == 'ST':
                # print("Prime the target successfully")
                break
            else:
                # print(f"Prime {i} times, current state is {now_state}")
                i += 1
                time.sleep(0.1)
                continue
        
        # step 2: execute the victim thread
        # The victim thread executes its branch with taken or not-taken 
        # and changes the state of the saturating counter accordingly depending on the executed program.
        now_state = self.step(now_state, victim_thread)

        # step 3: Probe the target
        # To distinguish the execution result of the victim’s branch, 
        # the probe vector used for spy must be the opposite of the prime vector used for initialization. 
        # Therefore, successive not-taken branches are executed. 
        # It stops when the prediction is hit with the execution and the attacker counts the number of the steps in executing the branch.
        if input_sequence is None:
            # !!! WE MUST COUNT THE NUMBER OF STEPS
            count = 0
            while True:
                now_state = self.step(now_state, PSCInputs.NT)
                count += 1
                if self.states[now_state].outputs == "NT":
                    # print(f"Probe the target successfully, the number of steps is {count}")

                    return count
                else:
                    # print(f"Probe {count} times, current state is {now_state}")
                    time.sleep(0.1)
                    continue
                
        else:
            self.run(now_state, input_sequence)
            print(f"Probe the target successfully, the current state is {self.run_history[-1]}")
            return len(self.run_history)
        



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



       