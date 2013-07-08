import numpy as np

def load_config(filename):
    conf = {}
    execfile(filename, conf)
    conf['global_conf']['area'] = np.asarray(conf['global_conf']['area'])
    return conf
