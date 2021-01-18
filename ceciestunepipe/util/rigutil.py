# tools to read/write the metadata of the rig.
# mainly, about the rig.json file
import json

'''
example of a rig_par dictionary:
rig_par = {'chan': {'ttl': {'trig_perceptron': 'DIN-01', 'trig_recording': 'DIN-00'}, 
                   'adc': {'microphone_0': 'ADC-00'},
                   'port': {'probe_0': 'A-'}
                 },
           'probe': {'probe_0': {'model': 'a1x32-edge-5mm-20-177_H32',
                                'serial': '768b',
                                'headstage': 'intan32-h32'}}
          }
'''
def get_rig_par(exp_struct: dict) -> dict:
    rig_par_file = exp_struct['files']['rig']
    with open(rig_par_file, 'r') as fp:
        rig_par = json.load(fp)
    return rig_par

def lookup_signal(rig_par_dict: dict, signal_name: str) -> tuple:
    chan_dict = rig_par_dict['chan']
    for ch_type, chans in chan_dict.items():
        found = [channel for signal, channel in chans.items() if signal == signal_name]
        if len(found) > 0:
            found = found[0]
            break
        else:
            ch_type = None
            found = ''
    return ch_type, found

def get_probe_port(exp_struct:dict, selected_probe:str) -> str:
    # get the probe and the port where the probe was connected
    rig_par = get_rig_par(exp_struct)
    probe_port = rig_par['chan']['port'][selected_probe].strip('-')
    return probe_port