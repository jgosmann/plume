import numpy as np

def load_config(filename):
    conf = {}
    execfile(filename, conf)
    conf['area'] = np.asarray(conf['area'])
    return conf
