# from config import attack
from SaturatingCounters import PrSaturatingCounter, PrivacySaturatingCounter
from markov_chain import state, PSCInputs, PSCOutputs
# import tqdm
import json
import os

def calculate_sn_nt_sn(m, p):
    return 1 - m + m * (1 - p)

def calculate_sn_nt_wn1(m, p):
    return m * p

def calculate_sn_t_sn(m, p):
    return 1 - m + m * p
def calculate_sn_t_wn1(m, p):
    return m * (1 - p)

def calculate_st_nt_wt1(m, p):
    return m * (1 - p)

def calculate_st_nt_st(m, p):
    return 1 - m + m * p
def calculate_st_t_st(m, p):
    return 1 - m + m * (1 - p)

def calculate_st_t_wt1(m, p):
    return m * p

def get_2bit_psc_model(probability_parameter):
    return PrSaturatingCounter(states={"SN": state("SN", PSCOutputs.NT),
                                           "WN1": state("WN1", PSCOutputs.NT),
                                            "ST": state("ST", PSCOutputs.T),
                                           "WT1": state("WT1", PSCOutputs.T)},
                                probability_threshold=probability_parameter,
                                transition_dict={
                                    "SN": {PSCInputs.NT: {"SN": 1}, 
                                            PSCInputs.T: {"WN1": probability_parameter, "SN": 1-probability_parameter}},
                                    "WN1": {PSCInputs.NT: {"SN": probability_parameter, "WN1": 1-probability_parameter}, 
                                            PSCInputs.T: {"ST": probability_parameter, "WN1": 1-probability_parameter}},
                                    "ST": {PSCInputs.T: {"ST": 1}, 
                                            PSCInputs.NT: {"WT1": probability_parameter, "ST": 1-probability_parameter}},
                                    "WT1": {PSCInputs.T: {"ST": probability_parameter, "WT1": 1-probability_parameter}, 
                                            PSCInputs.NT: {"SN": probability_parameter, "WT1": 1-probability_parameter}},
                                })

def get_3bit_psc_model(probability_parameter):
    return PrSaturatingCounter(states = {
                                        "SN": state("SN",   PSCOutputs.NT),
                                        "WN1": state("WN1", PSCOutputs.NT),
                                        "WN2": state("WN2", PSCOutputs.NT),
                                        "WN3": state("WN3", PSCOutputs.NT),
                                        "ST": state("ST",   PSCOutputs.T),
                                        "WT1": state("WT1", PSCOutputs.T),
                                        "WT2": state("WT2", PSCOutputs.T),
                                        "WT3": state("WT3", PSCOutputs.T),        
                                    },

                                probability_threshold=probability_parameter,
                                transition_dict={
                                    "SN": {PSCInputs.NT: {"SN": 1}, 
                                            PSCInputs.T: {"WN1": probability_parameter, "SN": 1-probability_parameter}},
                                    "WN1": {PSCInputs.NT: {"SN": probability_parameter, "WN1": 1-probability_parameter}, 
                                            PSCInputs.T: {"WN2": probability_parameter, "WN1": 1-probability_parameter}},
                                    "WN2": {PSCInputs.NT: {"WN1": probability_parameter, "WN2": 1-probability_parameter}, 
                                            PSCInputs.T: {"WN3": probability_parameter, "WN2": 1-probability_parameter}},
                                    "WN3": {PSCInputs.NT: {"WN2": probability_parameter, "WN3": 1-probability_parameter}, 
                                            PSCInputs.T: {"ST": probability_parameter, "WN3": 1-probability_parameter}},
                                    "ST": {PSCInputs.T: {"ST": 1}, 
                                            PSCInputs.NT: {"WT1": probability_parameter, "ST": 1-probability_parameter}},
                                    "WT1": {PSCInputs.T: {"ST": probability_parameter, "WT1": 1-probability_parameter}, 
                                            PSCInputs.NT: {"WT2": probability_parameter, "WT1": 1-probability_parameter}},
                                    "WT2": {PSCInputs.T: {"WT1": probability_parameter, "WT2": 1-probability_parameter}, 
                                            PSCInputs.NT: {"WT3": probability_parameter, "WT2": 1-probability_parameter}},
                                    "WT3": {PSCInputs.T: {"WT2": probability_parameter, "WT3": 1-probability_parameter}, 
                                            PSCInputs.NT: {"SN": probability_parameter, "WT3": 1-probability_parameter}}
                                    })


def generate_test_victim_thread(number: int):
    '''
    Randomly generate the victim thread, such as ["NT", "T", "NT", "T", ...]

    @param number: the number of victim thread
    '''
    import random
    victim_thread = []
    for i in range(number):
        victim_thread.append(random.choice(["NT", "T"]))
    return victim_thread

def count_suceess_rate_and_probe_number(model, victim_thread_list):
    '''
    Count the success rate of the model

    @param model: the model to be tested
    @param victim_thread_list: the list of victim thread
    @return: the count number
    @return: the success rate
    '''
    success_count = 0
    count_list = []
    for victim_thread in victim_thread_list:
        count =  model.run_attack("ST", victim_thread)
        count_list.append(count)
        result = model.get_attack_strategy(count)
        if result == victim_thread:
            success_count += 1
        if success_count % 50 == 0:
            print(f'The success count is {success_count}')
    return count_list, success_count/len(victim_thread_list)

def statistics_probe_number(count_list):
    '''
    get the probablity distribution of the count number of probe
    '''
    from collections import Counter
    count_dict = Counter(count_list)
    # every count number divide the total number
    for key in count_dict.keys():
        count_dict[key] = count_dict[key] / len(count_list)
    return count_dict


def experiment1(probability_parameter, victim_thread_number = 10000):
    '''
    Compare the performance of different models with different parameters

    models: PrSaturatingCounter and PrivacySaturatingCounter of 2-bit and 3-bit PSC
    parameters: probability_parameter (we use the privacy_parameter with best practical performance 0.1)
    '''
    
    # generate the victim thread
    victim_thread_list = generate_test_victim_thread(victim_thread_number)


    print('PrSaturatingCounter')
    print('2-bit PSC')
    prsc2bit = get_2bit_psc_model(probability_parameter)
    count_number_psc2bit, success_rate_psc2bit = count_suceess_rate_and_probe_number(prsc2bit, victim_thread_list)
    print(f'The average success rate of PrSaturatingCounter with 2-bit PSC is {success_rate_psc2bit} when probability_parameter = {probability_parameter}')
    print('3-bit PSC')
    prsc3bit = get_3bit_psc_model(probability_parameter)
    count_number_psc3bit,success_rate_psc3bit = count_suceess_rate_and_probe_number(prsc3bit, victim_thread_list)
    print(f'The average success rate of PrSaturatingCounter with 3-bit PSC is {success_rate_psc3bit} when probability_parameter = {probability_parameter}')
    #statistics the probablity distribution of the count number of probe
    count_dict_psc2bit = statistics_probe_number(count_number_psc2bit)
    count_dict_psc3bit = statistics_probe_number(count_number_psc3bit)
    count_dict_psc2bit['success_rate'] = success_rate_psc2bit
    count_dict_psc3bit['success_rate'] = success_rate_psc3bit
    psc2bit_file_folder = f'/root/moore/jsonfile/psc2bit'
    psc3bit_file_folder = f'/root/moore/jsonfile/psc3bit'
    if not os.path.exists(psc2bit_file_folder):
        os.makedirs(psc2bit_file_folder)
    if not os.path.exists(psc3bit_file_folder):
        os.makedirs(psc3bit_file_folder)
    # save the file
    with open(f'{psc2bit_file_folder}/psc2bit_{probability_parameter}.json', 'w') as f:
        json.dump(count_dict_psc2bit, f)
    with open(f'{psc3bit_file_folder}/psc3bit_{probability_parameter}.json', 'w') as f:
        json.dump(count_dict_psc3bit, f)
    return success_rate_psc2bit, success_rate_psc3bit
def experiment2(probability_parameter, privacy_parameter = 0.1, victim_thread_number = 10000):
    '''
    Compare the performance of different models with different parameters

    models: PrSaturatingCounter and PrivacySaturatingCounter of 2-bit and 3-bit PSC
    parameters: probability_parameter (we use the privacy_parameter with best practical performance 0.1)
    '''
    
    # generate the victim thread
    victim_thread_list = generate_test_victim_thread(victim_thread_number)

    # 2-bit PSC

    # print('PrSaturatingCounter')
    # print('2-bit PSC')
    # prsc2bit = get_2bit_psc_model(probability_parameter)
    # count_number_psc2bit, success_rate_psc2bit = count_suceess_rate_and_probe_number(prsc2bit, victim_thread_list)
    # print(f'The average success rate of PrSaturatingCounter with 2-bit PSC is {success_rate_psc2bit} when probability_parameter = {probability_parameter}')
    # print('3-bit PSC')
    # prsc3bit = get_3bit_psc_model(probability_parameter)
    # count_number_psc3bit,success_rate_psc3bit = count_suceess_rate_and_probe_number(prsc3bit, victim_thread_list)
    # print(f'The average success rate of PrSaturatingCounter with 3-bit PSC is {success_rate_psc3bit} when probability_parameter = {probability_parameter}')

    # now privacy saturating counter
    print('PrivacySaturatingCounter')
    print('2-bit privacy PSC')
    prsc2bit = PrivacySaturatingCounter(states={"SN": state("SN", PSCOutputs.NT),
                                           "WN1": state("WN1", PSCOutputs.NT),
                                            "ST": state("ST", PSCOutputs.T),
                                           "WT1": state("WT1", PSCOutputs.T)},
                                probability_threshold=probability_parameter,
                                privacy_threshold=privacy_parameter,
                                transition_dict = {
                                "SN":  {
                                    PSCInputs.NT: {"SN": calculate_sn_nt_sn(probability_parameter, privacy_parameter),
                                                    "WN1": calculate_sn_nt_wn1(probability_parameter, privacy_parameter)},
                                    PSCInputs.T: {"SN": calculate_sn_t_sn(probability_parameter, privacy_parameter),
                                                    "WN1": calculate_sn_t_wn1(probability_parameter, privacy_parameter)}
                                },
                                "WN1": {
                                    PSCInputs.NT: {"SN": probability_parameter, "WN1": 1-probability_parameter}, 
                                    PSCInputs.T: {"ST": probability_parameter, "WN1": 1-probability_parameter}
                                },
                                "ST": {
                                    PSCInputs.NT: {"WT1": calculate_st_nt_wt1(probability_parameter, privacy_parameter),
                                                    "ST": calculate_st_nt_st(probability_parameter, privacy_parameter)},
                                    PSCInputs.T: {"ST": calculate_st_t_st(probability_parameter, privacy_parameter),
                                                    "WT1": calculate_st_t_wt1(probability_parameter, privacy_parameter)}
                                },
                                "WT1": {
                                    PSCInputs.NT: {"SN": probability_parameter, "WT1": 1-probability_parameter}, 
                                    PSCInputs.T: {"WT1": 1 - probability_parameter, "ST": probability_parameter}
                                }
                                })
    count_number_prsc2bit, success_rate_prsc2bit = count_suceess_rate_and_probe_number(prsc2bit, victim_thread_list)
    print(f'The average success rate of PrivacySaturatingCounter with 2-bit PSC is {success_rate_prsc2bit} when probability_parameter = {probability_parameter} and privacy_parameter = {privacy_parameter}')

    print('3-bit privacy PSC')
    prsc3bit = PrivacySaturatingCounter(states = {
                                        "SN": state("SN",   PSCOutputs.NT),
                                        "WN1": state("WN1", PSCOutputs.NT),
                                        "WN2": state("WN2", PSCOutputs.NT),
                                        "WN3": state("WN3", PSCOutputs.NT),
                                        "ST": state("ST",   PSCOutputs.T),
                                        "WT1": state("WT1", PSCOutputs.T),
                                        "WT2": state("WT2", PSCOutputs.T),
                                        "WT3": state("WT3", PSCOutputs.T),        
                                    },
                                probability_threshold=probability_parameter,
                                privacy_threshold=privacy_parameter,
                                transition_dict = {
                                    "SN":  {
                                        PSCInputs.NT: {"SN": calculate_sn_nt_sn(probability_parameter, privacy_parameter),
                                                        "WN1": calculate_sn_nt_wn1(probability_parameter, privacy_parameter)},
                                        PSCInputs.T: {"SN": calculate_sn_t_sn(probability_parameter, privacy_parameter),
                                                        "WN1": calculate_sn_t_wn1(probability_parameter, privacy_parameter)}
                                    },
                                    "WN1": {
                                        PSCInputs.NT: {"SN": probability_parameter, "WN1": 1-probability_parameter}, 
                                        PSCInputs.T: {"WN2": probability_parameter, "WN1": 1-probability_parameter}
                                    },
                                    "WN2": {
                                        PSCInputs.NT: {"WN1": probability_parameter, "WN2": 1-probability_parameter}, 
                                        PSCInputs.T: {"WN3": probability_parameter, "WN2": 1-probability_parameter}
                                    },
                                    "WN3": {
                                        PSCInputs.NT: {"WN2": probability_parameter, "WN3": 1-probability_parameter}, 
                                        PSCInputs.T: {"ST": probability_parameter, "WN3": 1-probability_parameter}
                                    },
                                    "ST": {
                                        PSCInputs.NT: {"WT1": calculate_st_nt_wt1(probability_parameter, privacy_parameter),
                                                        "ST": calculate_st_nt_st(probability_parameter, privacy_parameter)},
                                        PSCInputs.T: {"ST": calculate_st_t_st(probability_parameter, privacy_parameter),
                                                        "WT1": calculate_st_t_wt1(probability_parameter, privacy_parameter)}
                                    },
                                    "WT1": {
                                        PSCInputs.T: {"ST": probability_parameter, "WT1": 1-probability_parameter}, 
                                        PSCInputs.NT: {"WT2": probability_parameter, "WT1": 1-probability_parameter}
                                    },
                                    "WT2": {
                                        PSCInputs.T: {"WT1": probability_parameter, "WT2": 1-probability_parameter}, 
                                        PSCInputs.NT: {"WT3": probability_parameter, "WT2": 1-probability_parameter}
                                    },
                                    "WT3": {
                                        PSCInputs.T: {"WT2": probability_parameter, "WT3": 1-probability_parameter}, 
                                        PSCInputs.NT: {"SN": probability_parameter, "WT3": 1-probability_parameter}
                                    }
                                    })
    count_number_prsc3bit,success_rate_prsc3bit = count_suceess_rate_and_probe_number(prsc3bit, victim_thread_list)
    print(f'The average success rate of PrivacySaturatingCounter with 3-bit PSC is {success_rate_prsc3bit} when probability_parameter = {probability_parameter} and privacy_parameter = {privacy_parameter}')
    
    #statistics the probablity distribution of the count number of probe
    # count_dict_psc2bit = statistics_probe_number(count_number_psc2bit)
    # count_dict_psc3bit = statistics_probe_number(count_number_psc3bit)
    count_dict_prsc2bit = statistics_probe_number(count_number_prsc2bit)
    count_dict_prsc3bit = statistics_probe_number(count_number_prsc3bit)
    # add the success rate to the dictionary
    # count_dict_psc2bit['success_rate'] = success_rate_psc2bit
    # count_dict_psc3bit['success_rate'] = success_rate_psc3bit
    count_dict_prsc2bit['success_rate'] = success_rate_prsc2bit
    count_dict_prsc3bit['success_rate'] = success_rate_prsc3bit
    # save json file

    # construct file folder
    psc2bit_file_folder = f'/root/moore/jsonfile/psc2bit'
    psc3bit_file_folder = f'/root/moore/jsonfile/psc3bit'
    prsc2bit_file_folder = f'/root/moore/jsonfile/prsc2bit'
    prsc3bit_file_folder = f'/root/moore/jsonfile/prsc3bit'
    if not os.path.exists(psc2bit_file_folder):
        os.makedirs(psc2bit_file_folder)
    if not os.path.exists(psc3bit_file_folder):
        os.makedirs(psc3bit_file_folder)
    if not os.path.exists(prsc2bit_file_folder):
        os.makedirs(prsc2bit_file_folder)
    if not os.path.exists(prsc3bit_file_folder):
        os.makedirs(prsc3bit_file_folder)
    # save the file
    # with open(f'{psc2bit_file_folder}/psc2bit_{probability_parameter}.json', 'w') as f:
    #     json.dump(count_dict_psc2bit, f)
    # with open(f'{psc3bit_file_folder}/psc3bit_{probability_parameter}.json', 'w') as f:
    #     json.dump(count_dict_psc3bit, f)
    with open(f'{prsc2bit_file_folder}/prsc2bit_{probability_parameter}_{privacy_parameter}.json', 'w') as f:
        json.dump(count_dict_prsc2bit, f)
    with open(f'{prsc3bit_file_folder}/prsc3bit_{probability_parameter}_{privacy_parameter}.json', 'w') as f:
        json.dump(count_dict_prsc3bit, f)

    
    # return success_rate_psc2bit, success_rate_psc3bit, success_rate_prsc2bit, success_rate_prsc3bit
    return success_rate_prsc2bit, success_rate_prsc3bit



def expriment1_full(victim_thread_number = 1000):
    '''
    Test different probability parameters for psc
    '''
    probability_parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # construct the list to store
    psc2bit_list = []
    psc3bit_list = []
    for probability_parameter in probability_parameters:
        print(f'The probability parameter is {probability_parameter}')
        psc2bit, psc3bit = experiment1(probability_parameter, victim_thread_number=victim_thread_number)
        psc2bit_list.append(psc2bit)
        psc3bit_list.append(psc3bit)
        print(f'Experiment 1 for the probability parameter is {probability_parameter} is finished')
        print('The result is:')
        print('2-bit PSC')
        print(psc2bit_list)
        print('3-bit PSC')
        print(psc3bit_list)
def expriment2_full(victim_thread_number = 1000):
    '''
    Test different probability parameters for ex2
    '''
    probability_parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # construct the list to store
    psc2bit_list = []
    psc3bit_list = []
    prsc2bit_list = []
    prsc3bit_list = []
    # for privacy_parameter in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9]:
    for privacy_parameter in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        for probability_parameter in probability_parameters:
            print(f'The privacy parameter is {privacy_parameter}, and the probability parameter is {probability_parameter}')
            # psc2bit, psc3bit, prsc2bit, prsc3bit = experiment1(probability_parameter, privacy_parameter=privacy_parameter, victim_thread_number=victim_thread_number)
            prsc2bit, prsc3bit = experiment2(probability_parameter, privacy_parameter=privacy_parameter, victim_thread_number=victim_thread_number)

            # psc2bit_list.append(psc2bit)
            # psc3bit_list.append(psc3bit)
            prsc2bit_list.append(prsc2bit)
            prsc3bit_list.append(prsc3bit)
        print(f'Experiment 2 for the privacy parameter is {privacy_parameter} is finished')
        print('The result is:')
        # print('2-bit PSC')
        # print(psc2bit_list)
        # print('3-bit PSC')
        # print(psc3bit_list)
        print('2-bit privacy PSC')
        print(prsc2bit_list)
        print('3-bit privacy PSC')
        print(prsc3bit_list)
    # return psc2bit_list, psc3bit_list, prsc2bit_list, prsc3bit_list


if __name__ == "__main__":
    expriment1_full(victim_thread_number=1000)
    expriment2_full(victim_thread_number=1000)