#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import sys
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler as _ForkingPickler

from habitat.core.logging import logger

if sys.version_info[:2] < (3, 8):
    # pickle 5 backport
    try:
        import pickle5 as pickle
    except ImportError:
        import pickle  # type: ignore[no-redef]

        logger.warn(
            f"""Warning pickle v5 protocol not supported.
        Falling back to pickle version {pickle.HIGHEST_PROTOCOL}.
        pip install pickle5 or upgrade to Python 3.8 or greater
        for faster performance"""
        )

    class ForkingPickler5(pickle.Pickler):
        wrapped = _ForkingPickler
        loads = staticmethod(pickle.loads)

        @classmethod
        def dumps(cls, obj, protocol: int = -1):
            buf = io.BytesIO()
            cls(buf, protocol).dump(obj)
            return buf.getbuffer()

        def __init__(self, file, protocol: int = -1, **kwargs):
            super().__init__(file, protocol, **kwargs)
            self.dispatch_table = self.wrapped(
                file, protocol, **kwargs
            ).dispatch_table

else:
    import pickle

    ForkingPickler5 = _ForkingPickler


class ConnectionWrapper:
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects. Will use the Pickle5 backport if available."""

    def __init__(self, conn: Connection):
        self.conn: Connection = conn

    def send(self, obj):
        self._check_closed()
        self._check_writable()
        buf = io.BytesIO()
        ForkingPickler5(buf, -1).dump(obj)
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
