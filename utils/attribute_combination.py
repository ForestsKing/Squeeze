import copy
from functools import reduce
from typing import Iterable

import numpy as np


class AttributeCombination(dict):
    ANY = '__ANY__'

    def __init__(self, **kwargs):
        super().__init__(**{key: str(value) for key, value in kwargs.items()})
        self.__id = None
        self.non_any_keys = tuple()
        self.non_any_values = tuple()
        self.__is_terminal = False
        self.__update()

    def __update(self):
        self.__id = tuple((key, self[key]) for key in sorted(self.keys()))
        self.non_any_keys = tuple(_ for _ in sorted(self.keys()) if self[_] != self.ANY)
        self.non_any_values = tuple(self[_] for _ in sorted(self.keys()) if self[_] != self.ANY)
        self.__is_terminal = not any(self.ANY == value for value in self.values())

    def __eq__(self, other):
        return self.__id == other.__id

    def __lt__(self, other):
        return self.__id < other.__id

    def __le__(self, other):
        return self.__id <= other.__id

    def __hash__(self):
        return hash(self.__id)

    def __setitem__(self, key, value):
        super().__setitem__(key, str(value))
        self.__update()

    def __str__(self):
        return "&".join(f"{key}={value}" for key, value in zip(self.non_any_keys, self.non_any_values))

    @staticmethod
    def from_string(string: str, attribute_names):
        ret = AttributeCombination.get_root_attribute_combination(attribute_names)
        for pair in string.split("&"):
            if pair == "":
                continue
            key, value = pair.split("=")
            ret[key] = value
        return ret

    @staticmethod
    def batch_from_string(string: str, attribute_names):
        return frozenset({AttributeCombination.from_string(_, attribute_names) for _ in string.split(";")})

    @staticmethod
    def batch_to_string(sets: Iterable['AttributeCombination']):
        return ";".join(str(_) for _ in sets)

    def copy_and_update(self, other):
        o = copy.copy(self)
        o.update(other)
        o.__update()
        return o

    def index_dataframe(self, data):
        if len(self.non_any_values) == 0:
            return np.ones(len(data), dtype=np.bool)
        try:
            arr = np.zeros(shape=len(data), dtype=np.bool)
            if len(self.non_any_values) == 1:
                idx = data.index.get_loc(self.non_any_values[0])
            else:
                idx = data.index.get_loc(self.non_any_values)
            arr[idx] = True
            return arr
        except KeyError:
            return np.zeros(len(data), dtype=np.bool)

    @staticmethod
    def batch_index_dataframe(attribute_combinations, data):
        index = reduce(np.logical_or,
                       (_.index_dataframe(data) for _ in attribute_combinations),
                       np.zeros(len(data), dtype=np.bool))
        return index

    @staticmethod
    def get_root_attribute_combination(attribute_names):
        return AttributeCombination(**{key: AttributeCombination.ANY for key in attribute_names})

    def mask(self, keys):

        to_fill_keys = set(self.keys()) - set(keys)
        return self.copy_and_update({key: self.ANY for key in to_fill_keys})
