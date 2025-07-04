try:
    import psutil
except ImportError:  # Allow running without psutil installed
    psutil = None
import os
import gc
import numpy as np

def get_memory_usage():
    """Get the memory usage of the current process in MB."""
    if psutil is None:
        return 0.0
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert bytes to MB

def log_memory(label):
    """Log current memory usage with a descriptive label"""
    mem = get_memory_usage()
    print(f"MEMORY [{label}]: {mem:.2f} MB")
    return mem

def log_array_sizes(arrays_dict):
    """Log the sizes of arrays in a dictionary"""
    print("ARRAY SIZES:")
    for name, arr in arrays_dict.items():
        try:
            # Handle sparse matrices
            if hasattr(arr, 'getnnz') and hasattr(arr, 'shape'):
                print(f"  - {name}: shape={arr.shape}, nnz={arr.getnnz()}, type={type(arr)}")
            # Handle numpy arrays and other array-like objects
            elif hasattr(arr, 'itemsize') and hasattr(arr, 'size'):
                size_mb = (arr.itemsize * arr.size) / (1024 * 1024)
                print(f"  - {name}: shape={arr.shape}, size={size_mb:.2f} MB, type={type(arr)}")
            # Handle any other object
            else:
                print(f"  - {name}: type={type(arr)}")
        except Exception as e:
            print(f"  - {name}: error getting size - {e}")

def clear_memory():
    """Force garbage collection and report memory cleared"""
    before = get_memory_usage()
    gc.collect()
    after = get_memory_usage()
    print(f"MEMORY CLEARED: {before - after:.2f} MB, now at {after:.2f} MB")
    return after 