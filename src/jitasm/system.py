import ctypes
import mmap
import os
import platform
import sys

machine = platform.machine().lower()
if machine not in ("amd64", "x86_64"):
    raise RuntimeError("unsupported architecture: %s" % platform.machine())

# Windows

if sys.platform == "win32":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    MEM_COMMIT = 0x1000
    MEM_RESERVE = 0x2000
    MEM_RELEASE = 0x8000

    PAGE_READWRITE = 0x04
    PAGE_EXECUTE_READ = 0x20

    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]

    kernel32.VirtualFree.restype = ctypes.c_int
    kernel32.VirtualFree.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
    ]

    kernel32.VirtualProtect.restype = ctypes.c_int
    kernel32.VirtualProtect.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
    ]

    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []

    kernel32.FlushInstructionCache.restype = ctypes.c_int
    kernel32.FlushInstructionCache.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]

    def mmap_windows(size):
        assert type(size) is int
        ptr = kernel32.VirtualAlloc(
            None,
            size,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )

        if not ptr:
            err = ctypes.get_last_error()
            raise OSError(err, os.strerror(err))

        return ptr

    def unmap_windows(ptr, size):
        assert type(ptr) is int
        assert type(size) is int
        if not kernel32.VirtualFree(
            ctypes.c_void_p(ptr),
            0,
            MEM_RELEASE,
        ):
            err = ctypes.get_last_error()
            raise OSError(err, os.strerror(err))

    def set_mem_rx_windows(ptr, size):
        assert type(ptr) is int
        assert type(size) is int
        old_protect = ctypes.c_ulong()

        if not kernel32.VirtualProtect(
            ctypes.c_void_p(ptr),
            size,
            PAGE_EXECUTE_READ,
            ctypes.byref(old_protect),
        ):
            return -1

        if not kernel32.FlushInstructionCache(
            kernel32.GetCurrentProcess(),
            ctypes.c_void_p(ptr),
            size,
        ):
            return -1
        return 0

    def get_page_size_windows():
        return mmap.PAGESIZE

    memory_map = mmap_windows
    unmap = unmap_windows
    set_mem_rx = set_mem_rx_windows
    get_page_size = get_page_size_windows

# POSIX

elif os.name == "posix":
    libc = ctypes.CDLL(None, use_errno=True)

    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_size_t,
    ]

    libc.munmap.restype = ctypes.c_int
    libc.munmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]

    libc.mprotect.restype = ctypes.c_int
    libc.mprotect.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    def mmap_posix(size):
        assert type(size) is int
        ptr = libc.mmap(
            None,
            size,
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            -1,
            0,
        )

        if ptr == ctypes.c_void_p(-1).value:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

        return ptr

    def unmap_posix(ptr, size):
        assert type(ptr) is int
        assert type(size) is int
        if libc.munmap(
            ctypes.c_void_p(ptr),
            size,
        ) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

    def set_mem_rx_posix(ptr, size):
        assert type(ptr) is int
        assert type(size) is int
        return libc.mprotect(
            ctypes.c_void_p(ptr),
            size,
            mmap.PROT_READ | mmap.PROT_EXEC,
        )

    def get_page_size_posix():
        return mmap.PAGESIZE

    memory_map = mmap_posix
    unmap = unmap_posix
    set_mem_rx = set_mem_rx_posix
    get_page_size = get_page_size_posix

else:
    raise RuntimeError("unsupported platform: %s" % sys.platform)
