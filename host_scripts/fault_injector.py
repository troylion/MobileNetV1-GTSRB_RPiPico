"""
Author: Kristian Bailey, North Carolina A&T State University
Project: MobileNet GTSRB SWIFI Framework

TFLite Fault Injection Tool v2
===============================
Alternative implementation that directly accesses buffer data
without relying on internal FlatBuffer _tab API.

Requirements:
    pip install flatbuffers tflite numpy

Usage:
    Same as v1 - set config and run
"""

import os
import sys
import numpy as np

try:
    import flatbuffers
    from tflite.Model import Model
    from tflite.Buffer import Buffer
except ImportError:
    print("Missing dependencies. Run:\n  pip install flatbuffers tflite numpy")
    sys.exit(1)


# ==============================================================================
#  CONFIG
# ==============================================================================

MODEL_PATH   = r"C:\Users\krist\Downloads\mobilenet_gtsrb.tflite"
OUTPUT_PATH  = "mobilenet_gtsrb_faulted.tflite"

INSPECT_ONLY = False  # If True, just list tensors and exit. Set to False to perform fault injection.

FAULT_TYPE   = "bit_flip"
TARGET_TENSOR_IDX = 5

BIT_FLIP_WEIGHT_IDX = None  # If None, random weight will be chosen
BIT_FLIP_BIT_POS    = 7

ZERO_FILTER_IDX     = 0
NOISE_STD           = 0.01
CLAMP_MIN           = -0.5
CLAMP_MAX           =  0.5

DTYPE = np.int8  # Set to np.float32, np.float16, np.int8, etc. depending on the tensor's data type

RANDOM_SEED = 42


# ==============================================================================
#  CORE HELPERS - Alternative implementation
# ==============================================================================

def load_model(path: str):
    """Load .tflite file into a bytearray and parse the FlatBuffer."""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    model = Model.GetRootAsModel(buf, 0)
    print(f"[INFO] Loaded '{path}' ({len(buf):,} bytes)")
    print(f"       Subgraphs: {model.SubgraphsLength()} | Buffers: {model.BuffersLength()}")
    return buf, model


def save_model(buf: bytearray, path: str):
    """Write the (modified) bytearray back to disk."""
    with open(path, "wb") as f:
        f.write(buf)
    print(f"[INFO] Saved faulted model to '{path}'")


# TFLite type enum -> numpy dtype mapping
TFLITE_TYPE_TO_NUMPY = {
    0: np.float32,
    1: np.float16,
    2: np.int32,
    3: np.uint8,
    9: np.int8,
}


def get_tensor_dtype(model, tensor_idx: int):
    """Auto-detect the numpy dtype for a tensor from its TFLite type field."""
    subgraph = model.Subgraphs(0)
    tensor = subgraph.Tensors(tensor_idx)
    type_id = tensor.Type()
    dtype = TFLITE_TYPE_TO_NUMPY.get(type_id)
    if dtype is None:
        raise ValueError(f"Tensor {tensor_idx} has unsupported TFLite type: {type_id}")
    return dtype


def get_tensor_data_as_numpy(model, buf: bytearray, tensor_idx: int, dtype=None):
    """
    Extract tensor data directly using Buffer.DataAsNumpy() or manual byte extraction.
    Returns a copy of the weight array.
    
    If dtype is None or 'auto', auto-detects from the tensor's TFLite type.
    """
    if dtype is None or dtype == 'auto':
        dtype = get_tensor_dtype(model, tensor_idx)
    
    subgraph = model.Subgraphs(0)
    tensor   = subgraph.Tensors(tensor_idx)
    buffer   = model.Buffers(tensor.Buffer())
    data_len = buffer.DataLength()
    
    if data_len == 0:
        raise ValueError(f"Tensor {tensor_idx} has no weight data")
    
    # Extract raw bytes using the buffer's Data method
    raw_bytes = bytearray()
    for i in range(data_len):
        raw_bytes.append(buffer.Data(i))
    
    # Convert to numpy
    arr = np.frombuffer(bytes(raw_bytes), dtype=dtype).copy()
    shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
    return arr.reshape(shape), raw_bytes


def find_buffer_data_in_file(buf: bytearray, raw_bytes: bytearray) -> int:
    """
    Find the offset where the buffer data appears in the file.
    This is a fallback method when FlatBuffer API doesn't work.
    """
    # Search for the exact byte sequence
    search_bytes = bytes(raw_bytes[:min(32, len(raw_bytes))])  # Use first 32 bytes as signature
    
    offset = buf.find(search_bytes)
    if offset == -1:
        raise ValueError("Could not locate buffer data in file")
    
    return offset


# ==============================================================================
#  INSPECT
# ==============================================================================

def inspect_model(model, buf: bytearray):
    """Print all tensors that contain weight data."""
    subgraph = model.Subgraphs(0)
    print("\n" + "=" * 75)
    print(f"{'Idx':>4}  {'buf':>4}  {'dtype':<8}  {'bytes':>8}  {'shape':<28}  name")
    print("=" * 75)

    for i in range(subgraph.TensorsLength()):
        tensor   = subgraph.Tensors(i)
        buf_idx  = tensor.Buffer()
        buffer   = model.Buffers(buf_idx)
        data_len = buffer.DataLength()

        if data_len == 0:
            continue

        shape    = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
        name     = tensor.Name().decode("utf-8") if tensor.Name() else "(no name)"
        dtype_id = tensor.Type()
        dtype_str = {0: "float32", 1: "float16", 2: "int32", 3: "uint8", 9: "int8"}.get(dtype_id, f"type{dtype_id}")

        print(f"{i:>4}  {buf_idx:>4}  {dtype_str:<8}  {data_len:>8}  {str(shape):<28}  {name}")

    print("=" * 75)
    print("\n[INFO] Only tensors with weight data are shown above.")
    print("[INFO] Set TARGET_TENSOR_IDX to one of these indices, then set INSPECT_ONLY=False.\n")


# ==============================================================================
#  FAULT INJECTION
# ==============================================================================

def fault_bit_flip(buf: bytearray, model, tensor_idx: int,
                   experiment_mode=4, dtype=np.float32):
    """Flip a single bit in one weight value based on the experiment mode."""
    
    # Get the weight data
    arr, raw_bytes = get_tensor_data_as_numpy(model, buf, tensor_idx, dtype)
    
    itemsize = np.dtype(dtype).itemsize
    num_weights = len(raw_bytes) // itemsize
    total_bits = itemsize * 8
    
    arr_flat = arr.flatten()
    
    # Filter candidate weights depending on the experiment
    if experiment_mode == 1:
        valid_indices = np.where(arr_flat > 0)[0]
    elif experiment_mode == 2:
        valid_indices = np.where(arr_flat > 0)[0]
    elif experiment_mode == 3:
        valid_indices = np.where(arr_flat < 0)[0]
    else:
        valid_indices = np.arange(num_weights)
        
    if len(valid_indices) == 0:
        raise ValueError(f"No weights match criteria for experiment mode {experiment_mode}")
        
    weight_idx = int(np.random.choice(valid_indices))
    
    # Find where this data is in the file
    data_offset = find_buffer_data_in_file(buf, raw_bytes)
    
    # Calculate byte offset of the specific weight
    byte_offset = data_offset + weight_idx * itemsize
    # Read original value
    original_bytes = bytes(buf[byte_offset : byte_offset + itemsize])
    original_val = np.frombuffer(original_bytes, dtype=dtype)[0]

    # Select bit position based on experiment
    if experiment_mode == 1:
        bit_pos = int(np.random.randint(0, total_bits - 1)) # Exclude MSB
    elif experiment_mode == 2 or experiment_mode == 3:
        bit_pos = int(total_bits - 1)                       # Force MSB
    else:
        bit_pos = int(np.random.randint(0, total_bits))     # Any bit

    bit_in_byte = bit_pos % 8
    byte_in_weight = bit_pos // 8
    
    # Flip the bit
    buf[byte_offset + byte_in_weight] ^= (1 << bit_in_byte)
    
    # Read new value
    new_bytes = bytes(buf[byte_offset : byte_offset + itemsize])
    new_val = np.frombuffer(new_bytes, dtype=dtype)[0]
    
    subgraph = model.Subgraphs(0)
    tensor = subgraph.Tensors(tensor_idx)
    tensor_name = tensor.Name().decode("utf-8") if tensor.Name() else "(no name)"
    
    print(f"\n[FAULT] bit_flip")
    print(f"  Tensor      : {tensor_idx} — {tensor_name}")
    print(f"  Weight      : index {weight_idx} / {num_weights - 1}")
    print(f"  Bit         : {bit_pos} (byte {byte_in_weight}, bit-in-byte {bit_in_byte})")
    print(f"  File offset : {byte_offset} (0x{byte_offset:X})")
    print(f"  Before      : {original_val}")
    print(f"  After       : {new_val}")
    
    return {
        "tensor_idx": tensor_idx,
        "weight_idx": int(weight_idx),
        "bit_pos": int(bit_pos),
        "original_val": int(original_val),
        "new_val": int(new_val),
    }


def fault_zero_filter(buf: bytearray, model, tensor_idx: int,
                      filter_idx: int, dtype=np.float32):
    """Zero out an entire filter."""
    
    arr, raw_bytes = get_tensor_data_as_numpy(model, buf, tensor_idx, dtype)
    
    if arr.ndim < 2:
        raise ValueError("Tensor has fewer than 2 dimensions")
    if filter_idx >= arr.shape[0]:
        raise ValueError(f"filter_idx {filter_idx} out of range (shape[0]={arr.shape[0]})")
    
    original_nonzero = np.count_nonzero(arr[filter_idx])
    arr[filter_idx] = 0
    
    # Find and update the data in the file
    data_offset = find_buffer_data_in_file(buf, raw_bytes)
    new_bytes = arr.astype(dtype).tobytes()
    buf[data_offset : data_offset + len(new_bytes)] = new_bytes
    
    subgraph = model.Subgraphs(0)
    tensor = subgraph.Tensors(tensor_idx)
    tensor_name = tensor.Name().decode("utf-8") if tensor.Name() else "(no name)"
    
    print(f"\n[FAULT] zero_filter")
    print(f"  Tensor  : {tensor_idx} — {tensor_name}")
    print(f"  Filter  : {filter_idx} (shape={list(arr[filter_idx].shape)})")
    print(f"  Zeroed  : {original_nonzero} weights → 0")


def fault_noise(buf: bytearray, model, tensor_idx: int,
                std: float = 0.01, dtype=np.float32):
    """Add Gaussian noise to all weights."""
    
    arr, raw_bytes = get_tensor_data_as_numpy(model, buf, tensor_idx, dtype)
    
    noise = np.random.normal(0, std, arr.shape).astype(dtype)
    arr += noise
    
    # Update file
    data_offset = find_buffer_data_in_file(buf, raw_bytes)
    new_bytes = arr.astype(dtype).tobytes()
    buf[data_offset : data_offset + len(new_bytes)] = new_bytes
    
    subgraph = model.Subgraphs(0)
    tensor = subgraph.Tensors(tensor_idx)
    tensor_name = tensor.Name().decode("utf-8") if tensor.Name() else "(no name)"
    
    print(f"\n[FAULT] noise")
    print(f"  Tensor    : {tensor_idx} — {tensor_name}")
    print(f"  Weights   : {arr.size} values perturbed")
    print(f"  Noise     : N(0, std={std})")
    print(f"  |noise| max: {np.abs(noise).max():.6f}  mean: {np.abs(noise).mean():.6f}")


def fault_clamp(buf: bytearray, model, tensor_idx: int,
                vmin: float, vmax: float, dtype=np.float32):
    """Clamp all weights to a range."""
    
    arr, raw_bytes = get_tensor_data_as_numpy(model, buf, tensor_idx, dtype)
    
    clipped = np.clip(arr, vmin, vmax)
    num_affected = np.sum((arr < vmin) | (arr > vmax))
    
    # Update file
    data_offset = find_buffer_data_in_file(buf, raw_bytes)
    new_bytes = clipped.astype(dtype).tobytes()
    buf[data_offset : data_offset + len(new_bytes)] = new_bytes
    
    subgraph = model.Subgraphs(0)
    tensor = subgraph.Tensors(tensor_idx)
    tensor_name = tensor.Name().decode("utf-8") if tensor.Name() else "(no name)"
    
    print(f"\n[FAULT] clamp")
    print(f"  Tensor   : {tensor_idx} — {tensor_name}")
    print(f"  Range    : [{vmin}, {vmax}]")
    print(f"  Clamped  : {num_affected} / {arr.size} weights")


# ==============================================================================
#  AUTOMATION ENTRY POINT
# ==============================================================================

_SEED_INITIALIZED = False

def run_injection(tensor_idx, experiment_mode=4, fault_type="bit_flip"):
    """
    Callable function for the master script to trigger a fault and regenerate the header.
    """
    global _SEED_INITIALIZED
    if RANDOM_SEED is not None and not _SEED_INITIALIZED:
        np.random.seed(RANDOM_SEED)
        import random
        random.seed(RANDOM_SEED)
        _SEED_INITIALIZED = True

    buf, model = load_model(MODEL_PATH)

    mode_note = f" [Experiment Mode: {experiment_mode}]"
    print(f"[INFO] Injecting fault on tensor {tensor_idx}...{mode_note}")

    if fault_type == "bit_flip":
        result = fault_bit_flip(
            buf,
            model,
            tensor_idx,
            experiment_mode=experiment_mode,
            dtype=DTYPE,
        )
    else:
        print(f"[ERROR] Unknown FAULT_TYPE '{fault_type}'")
        sys.exit(1)

    # 1. Save the .tflite binary
    save_model(buf, OUTPUT_PATH)

    # 2. Automatically generate the C-array header file!
    header_path = "model_faulted_data.h"
    print(f"[INFO] Automatically generating C header file: {header_path}")
    
    with open(header_path, "w") as f:
        f.write("#include <cstdint>\n\n")
        f.write("alignas(16) const unsigned char mobilenet_gtsrb_faulted_tflite[] = {\n  ")
        for i, byte in enumerate(buf):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n  ")
        f.write("\n};\n")
        f.write(f"unsigned int mobilenet_gtsrb_faulted_tflite_len = {len(buf)};\n")

    print("[DONE] Header generated!")
    return result

if __name__ == "__main__":
    # If run directly for testing, run a dummy injection
    print("This script is meant to be imported. Testing single run...")
    run_injection(5, experiment_mode=4)