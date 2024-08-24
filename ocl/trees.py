import copy
import dataclasses
from collections import OrderedDict, abc
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

import torch

Tree = Union[Dict, List, Tuple]

def get_tree_element(d: Tree, path: List[str]):
    """Get element of a tree."""
    next_element = d

    for next_element_name in path:
        if isinstance(next_element, abc.Mapping) and next_element_name in next_element:
            next_element = next_element[next_element_name]
        elif hasattr(next_element, next_element_name):
            next_element = getattr(next_element, next_element_name)
        elif isinstance(next_element, (list, tuple)) and next_element_name.isnumeric():
            next_element = next_element[int(next_element_name)]
        else:
            try:
                next_element = getattr(next_element, next_element_name)
            except AttributeError:
                msg = f"Trying to access path {'.'.join(path)}, "
                if isinstance(next_element, abc.Mapping):
                    msg += f"but element {next_element_name} is not among keys {next_element.keys()}"
                elif isinstance(next_element, (list, tuple)):
                    msg += f"but cannot index into list with {next_element_name}"
                else:
                    msg += (
                        f"but element {next_element_name} cannot be used to access attribute of "
                        f"object of type {type(next_element)}"
                    )
                raise ValueError(msg)
    return next_element