__author__ = 'max'

import lasagne


__all__ = [
    "get_all_params_by_name",
]


def get_all_params_by_name(layer, name=None, **tags):
    # tags['trainable'] = tags.get('trainable', True)
    # tags['regularizable'] = tags.get('regularizable', True)
    params = lasagne.layers.get_all_params(layer, **tags)
    if name is None:
        return params
    else:
        name_set = set(name) if isinstance(name, list) else set([name, ])
        return [param for param in params if param.name in name_set]
