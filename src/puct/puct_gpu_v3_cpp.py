import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
from numba import cuda


HERE = Path(__file__).resolve().parent
CU_PATH = HERE / "puct_gpu_v3_cpp.cu"
BUILD_DIR = Path(os.environ.get("PUCT_V3_CPP_BUILD_DIR", "/tmp/puct_gpu_v3_cpp"))


def _device_ptr(arr):
    return int(arr.device_ctypes_pointer.value)


def _stream_handle():
    try:
        handle = cuda.default_stream().handle
        return int(getattr(handle, "value", handle) or 0)
    except Exception:
        return 0


def _gpu_arch():
    try:
        dev = cuda.get_current_device()
        cc = dev.compute_capability
        return f"sm_{int(cc[0])}{int(cc[1])}"
    except Exception:
        return os.environ.get("PUCT_V3_CPP_ARCH", "sm_80")


def _source_hash(arch):
    h = hashlib.sha256()
    h.update(CU_PATH.read_bytes())
    h.update(arch.encode("utf-8"))
    return h.hexdigest()[:16]


def _compile_so():
    arch = os.environ.get("PUCT_V3_CPP_ARCH", _gpu_arch())
    digest = _source_hash(arch)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    so_path = BUILD_DIR / f"puct_gpu_v3_cpp_{arch}_{digest}.so"
    if so_path.exists():
        return so_path

    conda_prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    include_dir = conda_prefix / "targets" / "x86_64-linux" / "include"
    lib_dir = conda_prefix / "lib"
    cmd = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        f"-arch={arch}",
        "-o",
        str(so_path),
        str(CU_PATH),
    ]
    if include_dir.exists():
        cmd.extend(["-I", str(include_dir)])
    if lib_dir.exists():
        cmd.extend(["-L", str(lib_dir)])
    cmd.append("-lcudart")
    subprocess.run(cmd, check=True)
    return so_path


_LIB = None


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB

    lib = ctypes.CDLL(str(_compile_so()))
    c_void = ctypes.c_void_p
    c_int = ctypes.c_int
    c_float = ctypes.c_float
    lib.puct_select_winner_recalc_launch.argtypes = [
        c_float,
        c_float,
        c_float,
        c_int,
        c_void,
        c_void,
        c_void,
        c_void,
        c_void,
        c_void,
        c_void,
        c_void,
        c_void,
        c_void,
        c_void,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        c_void,
    ]
    lib.puct_select_winner_recalc_launch.restype = c_int
    lib.puct_cpp_cuda_error_string.argtypes = [c_int]
    lib.puct_cpp_cuda_error_string.restype = ctypes.c_char_p
    _LIB = lib
    return lib


def launch_select(
    d,
    cpuct=1.0,
    c_pw=1.0,
    alpha_pw=0.5,
    soft_winner=0,
):
    edge_child = d["edge_child"]
    trees = int(d["trees"])
    nodes = int(edge_child.shape[1])
    edges = int(edge_child.shape[2])
    warps = int(d["warps"])
    path_depth = int(d["out_path"].shape[2])
    code = _lib().puct_select_winner_recalc_launch(
        np.float32(cpuct),
        np.float32(c_pw),
        np.float32(alpha_pw),
        int(soft_winner),
        ctypes.c_void_p(_device_ptr(d["edge_child"])),
        ctypes.c_void_p(_device_ptr(d["edge_prior"])),
        ctypes.c_void_p(_device_ptr(d["edge_w"])),
        ctypes.c_void_p(_device_ptr(d["edge_n"])),
        ctypes.c_void_p(_device_ptr(d["edge_inflight"])),
        ctypes.c_void_p(_device_ptr(d["node_expand_inflight"])),
        ctypes.c_void_p(_device_ptr(d["node_expanded"])),
        ctypes.c_void_p(_device_ptr(d["node_count"])),
        ctypes.c_void_p(_device_ptr(d["out_selected"])),
        ctypes.c_void_p(_device_ptr(d["out_path"])),
        ctypes.c_void_p(_device_ptr(d["out_len"])),
        trees,
        nodes,
        edges,
        warps,
        path_depth,
        ctypes.c_void_p(_stream_handle()),
    )
    if code != 0:
        msg = _lib().puct_cpp_cuda_error_string(code).decode("utf-8", "replace")
        raise RuntimeError(f"C++ PUCT select launch failed: {msg} ({code})")


def is_available():
    try:
        _lib()
        return True
    except Exception:
        return False
