from importlib import import_module
from types import ModuleType

import numpy as np


def instantiate(
        module_name, class_name, args=(), kwargs={}, prefix_args=(),
        **additional_args):
    module = import_module(module_name)
    additional_args.update(kwargs)
    return getattr(module, class_name)(
        *(prefix_args + args), **additional_args)


def load_config(filename):
    conf = {}
    execfile(filename, conf)

    del conf['__builtins__']
    modules = [key for key, value in conf.iteritems()
               if isinstance(value, ModuleType)]
    for key in modules:
        del conf[key]

    if 'area' in conf:
        conf['area'] = np.asarray(conf['area'])
    return conf
