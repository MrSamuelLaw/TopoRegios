import asyncio
import numpy as np
from typing import List, Callable
from collections import deque


class R_TRIG():
    """Used to detect the leading edge of a signal
    as it goes from False to True"""

    def __init__(self, signal):
        self._values = deque((False, False), maxlen=2)
        self._bool = False

    def __call__(self, signal):
        self._values.append(signal)
        return (self._values[0] is False) and (self._values[1] is True)


class WatchableList(list):
    """A list with callbacks that are called on
    both append and remove. To add callbacks use
    subscribe and unsubscribe"""
    
    def __init__(self, *values):
        super().__init__()
        self.callbacks = []
        self.extend([*values])

    def subscribe(self, callback):
        self.callbacks.append(callback)

    def unsubscribe(self, callback):
        self.callbacks.remove(callback)
    
    def append(self, value):
        super().append(value)
        [cb(self) for cb in self.callbacks]

    def remove(self, value):
        super().remove(value)
        [cb(self) for cb in self.callbacks]
    

class GuiBinding():
    """Takes a list of objects and a list of callables that define
    how the data gets pushed into the object"""

    def __init__(self, objects: List[object], callbacks: List[Callable]):
        pass

    def __call__(self, caller, data):
        pass



class BoolTrigger:
    """This class wraps a boolean in functions
    for callback purposes where assignments may not
    be permitted, but functions are allowed"""

    def __init__(self, value: bool):
        self._value = value

    def __call__(self):
        return self._value
    
    def __eq__(self, boolean_value):
        return self._value == boolean_value

    def toggle(self):
        self._value = not self._value

    def true(self):
        self._value = True

    def false(self):
        self._value = False


class Counter:
    """Used to while loop safty
    the lam argument is a lambda that
    determines the counting behavior
    """

    def __init__(self, lam):
        self._lambda = lam
        self._count = 0

    def __set__(self, obj, value):
        self._count = value

    def __call__(self):
        self._count += 1
        return self._count


async def aprint(message):
    print(message)
    await asyncio.sleep(1E-9)


def multidim_xor(arr1, arr2) -> List[int]:
    """
    
    Explanation: 
    Using the view converts each entry of the array from an array of arrays 
    to an array of tuples with the origninal datatype.
    put more simply if we print arr1, assuming arr1 is a n,3 we would get
    np.array[
        np.array[1, 2, 3],
        np.array[4, 5, 6]...
    ]
    If we print arr1 viewed as dtype=[('', arr1.dtype)]*3
    we instead would get
    np.array[
        (1, 2, 3), <- tuple instead of np array
        (4, 5, 6)
    ]
    With tuples being a hashable type, we can use insections and 
    exclusive-or's on the underlying memory without allocating new memory.
    This allows for incredibly fast in place computations.
    """
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    mask = np.in1d(arr1_view, arr2_view)
    idx = np.nonzero(~mask)
    return idx


def multidim_intersect(arr1, arr2) -> List[int]:
    """
    Returns the intersections between two ndarrays
    Explanation: see multidim_xor
    
    """
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    mask = np.in1d(arr1_view, arr2_view)
    idx = np.nonzero(mask)
    return idx


def unique_rename(existing_names, base_name, ext):
    counter = Counter(lambda x: x + 1)
    while (counter() < 10):
        if not base_name in existing_names:
            return base_name
        base_name = base_name + ext