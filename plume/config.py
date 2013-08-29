from types import ModuleType
import numpy as np


class ConfObj(object):
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self, module, *prefix_args, **update_kwargs):
        kwargs = self.kwargs.copy()
        kwargs.update(update_kwargs)
        return getattr(module, self.cls)(*(prefix_args + self.args), **kwargs)


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
