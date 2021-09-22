#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import time
from contextlib import contextmanager
from timeit import default_timer
from contextlib import ContextDecorator

@contextmanager
def elapsed_timer():
    """
    Measure time elapsed in a block of code. Used for debugging.
    Taken from:
    https://stackoverflow.com/questions/7370801/measure-time-elapsed-in-python
    """
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

class TimeProfiler(ContextDecorator):
    def __init__(self, timer_name, timee=None, timer_prop=None):
        """
        - timer_prop: str The code that is used to access `self` when using the
          this as a method decorator.
        """
        self.timer_name = timer_name
        self.timer_prop = timer_prop
        if timee is not None:
            self.add_time_f = timee.timer.add_time
        else:
            self.add_time_f = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __call__(self, f):
        def wrapper(*args):
            other_self = args[0]
            if self.timer_prop is not None:
                self.add_time_f = eval(f"other_self.{self.timer_prop}.timer.add_time")
            else:
                self.add_time_f = other_self.timer.add_time
            return f(*args)
        return super().__call__(wrapper)

    def __exit__(self, *exc):
        elapsed = time.time() - self.start_time
        self.add_time_f(self.timer_name, elapsed)
        return False


class TimeProfilee:
    def __init__(self):
        self.clear()

    def add_time(self, timer_name, timer_val):
        self.timers[timer_name] += timer_val
        self.timer_call_count[timer_name] += 1

    def get_time(self, timer_name):
        return (self.timers[timer_name],
                self.timer_call_count[timer_name])

    def clear(self):
        self.timers = defaultdict(lambda: 0)
        self.timer_call_count = defaultdict(lambda: 0)