from types import ModuleType
import numpy as np


def load_config(filename):
    conf = {}
    execfile(filename, conf)

    del conf['__builtins__']
    modules = [key for key, value in conf.iteritems()
               if isinstance(value, ModuleType)]
    for key in modules:
        del conf[key]

    conf['global_conf']['area'] = np.asarray(conf['global_conf']['area'])
    return conf
