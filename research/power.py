"""Keep Windows awake only during the authorized calculation, then restore."""
from contextlib import contextmanager
import os


@contextmanager
def keep_awake():
    if os.name!='nt':
        yield
        return
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]
    kernel.SetThreadExecutionState.restype=ctypes.c_uint
    previous=kernel.SetThreadExecutionState(0x80000001)
    try:
        yield
    finally:
        if previous:
            kernel.SetThreadExecutionState(previous)
