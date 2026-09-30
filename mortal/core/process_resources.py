"""Explicit process-local resource settings; defaults leave OS policy unchanged."""
import ctypes
import logging
import os


def configure_windows_high_qos(enabled=False):
    """Opt this process into HighQoS without changing priority or CPU affinity.

    Windows may infer a lower QoS for a hidden background process even when its
    priority is Normal. This opt-in only disables execution-speed throttling;
    other power-throttling controls and the system power plan are preserved.
    """
    if not enabled or os.name != 'nt':
        return False
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG),
                    ('StateMask', wintypes.ULONG)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    for name in ('GetProcessInformation', 'SetProcessInformation'):
        function = getattr(kernel, name)
        function.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        function.restype = wintypes.BOOL
    process = kernel.GetCurrentProcess()
    state = PowerThrottlingState(1, 0, 0)
    # ProcessPowerThrottling=4, PROCESS_POWER_THROTTLING_EXECUTION_SPEED=1.
    if not kernel.GetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())
    state.ControlMask |= 1
    state.StateMask &= ~1
    if not kernel.SetProcessInformation(process, 4, ctypes.byref(state), ctypes.sizeof(state)):
        raise ctypes.WinError(ctypes.get_last_error())
    logging.getLogger(__name__).info(
        'Windows HighQoS enabled for pid=%s; priority and affinity unchanged', os.getpid())
    return True
