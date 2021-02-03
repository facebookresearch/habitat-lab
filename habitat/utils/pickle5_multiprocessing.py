import io
import sys
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler as _ForkingPickler
from typing import Dict

if sys.version_info[:2] < (3, 8):
    # pickle 5 backport
    try:
        import pickle5 as pickle
    except ImportError:
        import pickle  # type: ignore[no-redef]
else:
    import pickle


class ForkingPickler5(pickle.Pickler):
    _extra_reducers: Dict = dict()
    wrapped = _ForkingPickler
    loads = staticmethod(pickle.loads)

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


class ConnectionWrapper(object):
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects. Will use the Pickle5 backport if available."""

    def __init__(self, conn: Connection):
        self.conn: Connection = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler5(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        self._check_closed()
        self._check_readable()
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        if "conn" in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__, "conn"
            )
        )
