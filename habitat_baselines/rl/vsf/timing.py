import time
from collections import OrderedDict, deque


class AttrDict(OrderedDict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)


class AvgTime:
    def __init__(self, num_values_to_avg):
        if num_values_to_avg == "inf":
            self._sum = 0
            self._count = 0
        else:
            self.values = deque([], maxlen=num_values_to_avg)

        self.num_values_to_avg = num_values_to_avg

    def append(self, x):
        if self.num_values_to_avg == "inf":
            self._sum += x
            self._count += 1
        else:
            self.values.append(x)

    def value(self):
        if self.num_values_to_avg == "inf":
            avg_time = self._sum / self._count
        else:
            avg_time = sum(self.values) / max(1, len(self.values))

        return avg_time * 1e3

    def __str__(self):
        return f"{self.value():.4f}"


EPS = 1e-5


class TimingContext:
    def __init__(self, timer, key, additive=False, average=None):
        self._timer = timer
        self._key = key
        self._additive = additive
        self._average = average
        self._time_enter = None

    def __enter__(self):
        if self._key not in self._timer:
            if self._average is not None:
                self._timer[self._key] = AvgTime(
                    num_values_to_avg=self._average
                )
            else:
                self._timer[self._key] = 0

        self._time_enter = time.time()

    def __exit__(self, type_, value, traceback):
        time_passed = max(
            time.time() - self._time_enter, EPS
        )  # EPS to prevent div by zero

        if self._additive:
            self._timer[self._key] += time_passed
        elif self._average is not None:
            self._timer[self._key].append(time_passed)
        else:
            self._timer[self._key] = time_passed


class Timing(AttrDict):
    def timeit(self, key):
        return TimingContext(self, key)

    def add_time(self, key):
        return TimingContext(self, key, additive=True)

    def avg_time(self, key, average="inf"):
        return TimingContext(self, key, average=average)

    def __str__(self):
        s = ""
        i = 0
        for key, value in self.items():
            str_value = (
                f"{value:.4f}" if isinstance(value, float) else str(value)
            )
            s += f"{key}: {str_value}"
            if i < len(self) - 1:
                s += ", "
            i += 1
        return s
