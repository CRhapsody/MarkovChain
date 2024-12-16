# config.py
import argparse
from markov_chain import state, PSCInputs
from SaturatingCounters import NonPrSaturatingCounters, PrSaturatingCounter, PrivacySaturatingCounter
import json
import csv
def get_config():
    parser = argparse.ArgumentParser(description="Configuration for Saturating Counters")

    # General Parameters
    parser.add_argument('--model_type', type=str, choices=['non_pr_saturating_counters', 'pr_saturating_counter', 'privacy_saturating_counter'], 
                        default='pr_saturating_counter', help='Type of Saturating Counters Model')
    parser.add_argument('--m', type=float, default=0.5, help='Probability threshold (m)')
    parser.add_argument('--p', type=float, default=0.5, help='Privacy threshold (p)')
    
    # States and Transitions
    parser.add_argument('--states', type=str, nargs='+', default=["SN", "WN1", "WN2", "WN3", "ST", "WT1", "WT2", "WT3"], 
                        help='List of states') # need construct state dict or list
    parser.add_argument('--transition_dict', type=str, help='Path to JSON file containing transition dictionary')
    # parser.add_argument('--transition_matrix', type=str, default=None, help='Path to CSV file containing transition matrix')

    # Attack Parameters
    parser.add_argument('--initial_state', type=str, default='SN', help='Initial state for the attack')
    parser.add_argument('--victim_thread', type=str, choices=['NT', 'T'], default='T', help='Victim thread input')
    parser.add_argument('--input_sequence', type=str, nargs='+', default=None, help='List of inputs for the attack')

    args = parser.parse_args()
    return args

def initial_model(args):
    '''
    Initialize the Saturating Counters model
    '''
    # states = [state(name, PSCInputs(output)) for name, output in zip(args.states, ['NT']*4 + ['T']*4)]
    # according state name to construct state object, if state name has NT, then output is NT, otherwise output is T
    states = [state(name, PSCInputs.NT if 'N' in name else PSCInputs.T) for name in args.states]
    # read transition dictionary from file
    with open(args.transition_dict, 'r') as f:
        transition_dict = json.load(f)
    
    if args.model_type == 'pr_saturating_counter':
        return PrSaturatingCounter(states, args.m, transition_dict)
    elif args.model_type == 'non_pr_saturating_counters':
        return NonPrSaturatingCounters(states, transition_dict)
    elif args.model_type == 'privacy_saturating_counter':
        return PrivacySaturatingCounter(states, args.m, args.p, transition_dict)
    else:
        raise ValueError("Invalid model type")



