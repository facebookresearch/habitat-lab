import io
import multiprocessing
import multiprocessing.connection as mpc
import sys
from multiprocessing.reduction import AbstractReducer, ForkingPickler
from typing import Dict

# This patches the multiprocessing connection to serialize / deserialize pickle5
if sys.version_info[:2] <= (3, 8):
    # pickle 5 backport
    import pickle5

    class ForkingPickler5(pickle5.Pickler):
        _extra_reducers: Dict = dict()
        wrapped = mpc._ForkingPickler  # type: ignore
        loads = staticmethod(pickle5.loads)

        @classmethod
        def register(cls, type, reduce):  # noqa: A002
            """Register a reduce function for a type."""
            cls._extra_reducers[type] = reduce

        @classmethod
        def dumps(cls, obj, protocol=-1):
            buf = io.BytesIO()
            cls(buf, protocol).dump(obj)
            return buf.getbuffer()

        def __init__(self, file, protocol=-1, **kwargs):
            super().__init__(file, protocol, **kwargs)
            self.dispatch_table = self.wrapped(
                file, protocol, **kwargs
            ).dispatch_table
            self.dispatch_table.update(self._extra_reducers)


else:

    class ForkingPickler5(ForkingPickler):
        @classmethod
        def dumps(cls, obj, protocol=-1):
            return super().dumps(obj, protocol)


def dump(obj, file, protocol=-1):
    ForkingPickler5(file, protocol).dump(obj)


class Pickle5Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler5
    register = ForkingPickler5.register
    dump = dump

    def __init__(self, *args):
        super().__init__(*args)


mpc._ForkingPickler = ForkingPickler5  # type: ignore
multiprocessing.context._default_context.reducer.ForkingPickler = ForkingPickler5  # type: ignore
multiprocessing.context._default_context.reducer.dump = dump  # type: ignore
