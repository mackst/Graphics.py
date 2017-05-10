##########################################################################
#
# Copyright (C) 2017 by Shi Chi
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
#
##########################################################################

# from ctypes import (pythonapi, c_void_p, py_object)

from cffi import FFI
_ffi = FFI()
_ffi.cdef('long __stdcall GetWindowLongA(void* hWnd, int nIndex);')
_lib = _ffi.dlopen('User32.dll')


def getInstance(hWnd):
    return _lib.GetWindowLongA(_ffi.cast('void*', hWnd), -6)  # GWL_HINSTANCE

# def getWinHandles(winId):
#     pythonapi.PyCObject_AsVoidPtr.restype = c_void_p
#     pythonapi.PyCObject_AsVoidPtr.argtypes = [py_object]
#
#     hwnd = pythonapi.PyCObject_AsVoidPtr(winId)
#     hinstance = getInstance(hwnd)
#
#     return hwnd, hinstance