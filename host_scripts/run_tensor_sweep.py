"""
Tensor Sweep Campaign
======================
Runs faults on EVERY weight tensor in the model, scaled to 5%% of each
tensor's weight count (minimum 30 per tensor for statistical validity).
Produces a vulnerability heat map. Uses multiple Picos in parallel.

Usage:
    $env:PICO_PORTS="COM10,COM11,COM12,COM13"
    python run_tensor_sweep.py

Environment variables:
    FAULT_PERCENT       Percentage of weights to fault per tensor (default: 5)
    FAULT_FLOOR         Minimum faults per tensor (default: 30)
    PICO_PORTS          Comma-separated COM ports
    SKIP_FLASH          Skip compile+flash (default: "1")
"""

import csv
import json
import os
import queue
import re
import subprocess
import threading
import time
import traceback

import numpy as np
import serial
from serial.tools import list_ports

import fault_injector

# ──────────────────────── CONFIGURATION ────────────────────────

FAULT_PERCENT = float(os.environ.get("FAULT_PERCENT", "5"))  # % of weights
FAULT_FLOOR = int(os.environ.get("FAULT_FLOOR", "30"))       # minimum faults
EXPERIMENT_MODE = int(os.environ.get("EXPERIMENT_MODE", "4"))
BAUD_RATE = 115200
SERIAL_TIMEOUT_SEC = 120

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "campaign_results")

USERHOME = os.path.expanduser("~")
PICOTOOL_PATH = os.path.join(USERHOME, ".pico-sdk", "picotool", "2.2.0-a4", "picotool", "picotool.exe")

CAMPAIGN_ID = time.strftime("%Y%m%d_%H%M%S")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "sweep_checkpoint.json")
RESUME = os.environ.get("RESUME", "1") == "1"

# ──────────────────────── REGEX PATTERNS ────────────────────────

stop14_pattern = re.compile(r"^Class\s+14:\s+\d+\s*/\s*\d+\s+\(([^)]+)\)")
stop14_new_pattern = re.compile(r"^STOP14:\s+\d+/\d+\s+\(([^)]+)\)")

# ──────────────────────── THREAD SAFETY ────────────────────────

csv_lock = threading.Lock()
print_lock = threading.Lock()


def log(pico_id, msg):
    with print_lock:
        print(f"[Pico {pico_id}] {msg}")


# ──────────────────────── PICO DETECTION ────────────────────────

def find_all_pico_ports():
    manual = os.environ.get("PICO_PORTS", "")
    if manual:
        return [p.strip() for p in manual.split(",") if p.strip()]
    ports = []
    for port in list_ports.comports():
        desc = (port.description or "").lower()
        hwid = (port.hwid or "").lower()
        if any(tok in desc for tok in ["pico", "usb serial", "cdc"]) or "2e8a" in hwid:
            ports.append(port.device)
    return sorted(ports)


# ──────────────────────── DISCOVER WEIGHT TENSORS ────────────────────────

def discover_weight_tensors(model, buf):
    """Find all tensors that have weight data and their properties."""
    subgraph = model.Subgraphs(0)
    tensors = []

    DTYPE_MAP = {0: "float32", 1: "float16", 2: "int32", 3: "uint8", 9: "int8"}

    for i in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(i)
        buf_idx = tensor.Buffer()
        buffer = model.Buffers(buf_idx)
        data_len = buffer.DataLength()

        if data_len == 0:
            continue

        shape = [tensor.Shape(j) for j in range(tensor.ShapeLength())]
        name = tensor.Name().decode("utf-8") if tensor.Name() else "(no name)"
        type_id = tensor.Type()
        dtype_str = DTYPE_MAP.get(type_id, f"type{type_id}")
        np_dtype = fault_injector.TFLITE_TYPE_TO_NUMPY.get(type_id, np.int8)

        # Classify tensor type
        if len(shape) == 1:
            category = "bias"
        elif len(shape) == 4 and shape[0] == 1 and shape[1] == 3 and shape[2] == 3:
            category = "depthwise"
        elif len(shape) == 4 and shape[1] == 1 and shape[2] == 1:
            category = "pointwise"
        elif len(shape) == 4 and shape[1] == 3 and shape[2] == 3 and shape[0] > 1:
            category = "conv_input"
        elif len(shape) == 2:
            category = "fc"
        else:
            category = "other"

        itemsize = np.dtype(np_dtype).itemsize
        num_weights = data_len // itemsize
        num_faults = max(FAULT_FLOOR, int(np.ceil(num_weights * FAULT_PERCENT / 100)))

        tensors.append({
            "idx": i,
            "name": name,
            "shape": shape,
            "dtype": dtype_str,
            "np_dtype": np_dtype,
            "bytes": data_len,
            "num_weights": num_weights,
            "num_faults": num_faults,
            "category": category,
        })

    return tensors


# ──────────────────────── FAULT PRE-GENERATION ────────────────────────

def pregenerate_faults_for_tensor(model, buf, tensor_info, num_faults, experiment_mode):
    """Pre-generate fault parameters for a single tensor."""
    tensor_idx = tensor_info["idx"]
    dtype = tensor_info["np_dtype"]
    _, raw_bytes = fault_injector.get_tensor_data_as_numpy(model, buf, tensor_idx, dtype)

    itemsize = np.dtype(dtype).itemsize
    total_bits = itemsize * 8
    arr = np.frombuffer(bytes(raw_bytes), dtype=dtype).copy()
    arr_flat = arr.flatten()

    if experiment_mode == 1 or experiment_mode == 2:
        valid_indices = np.where(arr_flat > 0)[0]
    elif experiment_mode == 3:
        valid_indices = np.where(arr_flat < 0)[0]
    else:
        valid_indices = np.arange(len(raw_bytes) // itemsize)

    if len(valid_indices) == 0:
        return []

    faults = []
    for i in range(num_faults):
        rand_weight = int(np.random.choice(valid_indices))

        if experiment_mode == 1:
            actual_bit = int(np.random.randint(0, total_bits - 1))
        elif experiment_mode == 2 or experiment_mode == 3:
            actual_bit = total_bits - 1
        else:
            actual_bit = int(np.random.randint(0, total_bits))

        byte_offset = rand_weight * itemsize + (actual_bit // 8)
        bit_in_byte = actual_bit % 8

        weight_start = (byte_offset // itemsize) * itemsize
        orig_bytes = bytearray(raw_bytes[weight_start:weight_start + itemsize])
        before_value = np.frombuffer(bytes(orig_bytes), dtype=dtype)[0]

        faulted_bytes = bytearray(orig_bytes)
        byte_within = byte_offset - weight_start
        faulted_bytes[byte_within] ^= (1 << bit_in_byte)
        after_value = np.frombuffer(bytes(faulted_bytes), dtype=dtype)[0]

        before_val = float(before_value) if itemsize > 1 else int(before_value)
        after_val = float(after_value) if itemsize > 1 else int(after_value)

        faults.append({
            "tensor_idx": tensor_idx,
            "fault_num": i,
            "num_faults": num_faults,
            "rand_weight": rand_weight,
            "actual_bit": actual_bit,
            "byte_offset": byte_offset,
            "bit_in_byte": bit_in_byte,
            "before_val": before_val,
            "after_val": after_val,
            "category": tensor_info["category"],
            "dtype": tensor_info["dtype"],
        })

    return faults


# ──────────────────────── CHECKPOINT ────────────────────────

def load_checkpoint():
    if not RESUME or not os.path.exists(CHECKPOINT_PATH):
        return set(), None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        completed = set()
        for key in data.get("completed", []):
            completed.add(key)
        return completed, data.get("campaign_id")
    except Exception:
        return set(), None


def save_checkpoint(completed_set):
    data = {
        "campaign_id": CAMPAIGN_ID,
        "completed": sorted(completed_set),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with csv_lock:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ──────────────────────── SYNC HELPER ────────────────────────

def sync_pico(ser, pico_id):
    ser.reset_input_buffer()
    time.sleep(0.3)
    ser.write(b"SYNC\n")
    deadline = time.time() + 15
    while time.time() < deadline:
        line_bytes = ser.readline()
        if not line_bytes:
            continue
        line = line_bytes.decode("utf-8", errors="replace").strip()
        if "READY_FOR_CMD" in line:
            return True
    return False


# ──────────────────────── PICO WORKER ────────────────────────

def pico_worker(pico_id, port, fault_queue, completed_set, completed_lock,
                summary_writer, summary_file):
    try:
        # Reboot
        subprocess.run([PICOTOOL_PATH, "reboot", "-f"], capture_output=True)
        time.sleep(4)

        log(pico_id, f"Connecting to {port}...")
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT_SEC)
        except serial.SerialException as e:
            log(pico_id, f"ERROR: Could not open {port}: {e}")
            return

        if not sync_pico(ser, pico_id):
            log(pico_id, "ERROR: Could not sync")
            ser.close()
            return

        # Set quiet mode
        ser.write(b"MODE:QUIET\n")
        deadline = time.time() + 5
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if "MODE:QUIET" in line:
                break

        log(pico_id, "Ready!")
        faults_done = 0

        while True:
            try:
                fault = fault_queue.get(timeout=2)
            except queue.Empty:
                with completed_lock:
                    if fault_queue.empty():
                        break
                continue

            fault_key = f"{fault['tensor_idx']}_{fault['fault_num']}"
            tensor_idx = fault["tensor_idx"]
            byte_offset = fault["byte_offset"]
            bit_in_byte = fault["bit_in_byte"]

            with completed_lock:
                total_done = len(completed_set)

            log(pico_id, f"T{tensor_idx}({fault['category']}) fault {fault['fault_num']+1}/{fault['num_faults']} "
                         f"[{total_done} total done]")

            try:
                # Wait for READY
                ready_deadline = time.time() + 30
                ready = False
                while time.time() < ready_deadline:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                    if "READY_FOR_CMD" in line:
                        ready = True
                        break

                if not ready:
                    log(pico_id, "WARN: Timeout. Requeueing.")
                    fault_queue.put(fault)
                    break

                # Send injection
                cmd = f"INJECT:{tensor_idx},{byte_offset},{bit_in_byte}\n"
                fault_start = time.time()
                ser.write(cmd.encode())

                # Read results
                final_accuracy = "ERROR"
                stop14_accuracy = "N/A"

                while True:
                    line_bytes = ser.readline()
                    if not line_bytes:
                        continue
                    line = line_bytes.decode("utf-8", errors="replace").strip()

                    if "Accuracy:" in line:
                        parts = line.split("Accuracy:")
                        if len(parts) > 1:
                            final_accuracy = parts[1].strip()

                    stop_match = stop14_pattern.match(line) or stop14_new_pattern.match(line)
                    if stop_match:
                        stop14_accuracy = stop_match.group(1)

                    if "TEST_COMPLETE" in line:
                        break

                elapsed = time.time() - fault_start

                # Write result
                with csv_lock:
                    summary_writer.writerow([
                        CAMPAIGN_ID,
                        fault["tensor_idx"],
                        fault["category"],
                        fault["dtype"],
                        fault["fault_num"] + 1,
                        fault["rand_weight"],
                        fault["actual_bit"],
                        fault["before_val"],
                        fault["after_val"],
                        final_accuracy,
                        stop14_accuracy,
                    ])
                    summary_file.flush()

                with completed_lock:
                    completed_set.add(fault_key)
                    save_checkpoint(completed_set)

                faults_done += 1
                log(pico_id, f"  -> Acc={final_accuracy} Stop14={stop14_accuracy} ({elapsed:.1f}s)")

            except serial.SerialException as e:
                log(pico_id, f"ERROR: {e}. Requeueing.")
                fault_queue.put(fault)
                break
            except Exception as e:
                log(pico_id, f"ERROR: {e}. Requeueing.")
                fault_queue.put(fault)
                break

        try:
            ser.close()
        except Exception:
            pass
        log(pico_id, f"Worker done. {faults_done} faults completed.")

    except Exception as e:
        log(pico_id, f"FATAL: {e}")
        traceback.print_exc()


# ──────────────────────── MAIN ────────────────────────

def main():
    global CAMPAIGN_ID

    pico_ports = find_all_pico_ports()
    if not pico_ports:
        print("[ERROR] No Pico ports found. Set PICO_PORTS=COM10,COM11,...")
        return

    print("\n========== TENSOR SWEEP CONFIG ==========")
    print(f"Picos             : {len(pico_ports)} -> {', '.join(pico_ports)}")
    print(f"Fault coverage    : {FAULT_PERCENT}% of weights per tensor (floor={FAULT_FLOOR})")
    print(f"Experiment mode   : {EXPERIMENT_MODE}")
    print(f"Resume            : {RESUME}")
    print(f"Campaign ID       : {CAMPAIGN_ID}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model and discover tensors
    print("\n[INFO] Loading model...")
    buf, model = fault_injector.load_model(fault_injector.MODEL_PATH)

    weight_tensors = discover_weight_tensors(model, buf)
    print(f"[INFO] Found {len(weight_tensors)} weight tensors")

    # Print tensor summary
    print(f"\n{'Idx':>4}  {'Category':<12}  {'Dtype':<8}  {'Weights':>8}  {'Faults':>7}  {'Shape':<28}  Name")
    print("=" * 100)
    for t in weight_tensors:
        print(f"{t['idx']:>4}  {t['category']:<12}  {t['dtype']:<8}  {t['num_weights']:>8}  {t['num_faults']:>7}  {str(t['shape']):<28}  {t['name']}")
    print("=" * 100)

    total_planned = sum(t['num_faults'] for t in weight_tensors)
    print(f"\n[INFO] Fault plan: {FAULT_PERCENT}% of weights per tensor (floor={FAULT_FLOOR})")
    print(f"[INFO] Total planned faults: {total_planned} across {len(weight_tensors)} tensors")

    # Pre-generate all faults
    print(f"\n[INFO] Pre-generating faults...")
    np.random.seed(fault_injector.RANDOM_SEED)

    all_faults = []
    for t in weight_tensors:
        faults = pregenerate_faults_for_tensor(model, buf, t, t['num_faults'], EXPERIMENT_MODE)
        all_faults.extend(faults)
        if faults:
            print(f"  Tensor {t['idx']:>3} ({t['category']:<12}): {len(faults):>5} faults  ({t['num_weights']} weights, {t['dtype']})")

    total_faults = len(all_faults)
    print(f"\n[INFO] Total faults to run: {total_faults}")

    # Load checkpoint
    completed_set, prev_id = load_checkpoint()
    completed_lock = threading.Lock()

    if completed_set and prev_id:
        CAMPAIGN_ID = prev_id
        print(f"[INFO] Resuming campaign {CAMPAIGN_ID}: {len(completed_set)} faults done")

    # Build work queue
    fault_queue = queue.Queue()
    for fault in all_faults:
        fault_key = f"{fault['tensor_idx']}_{fault['fault_num']}"
        if fault_key not in completed_set:
            fault_queue.put(fault)

    remaining = fault_queue.qsize()
    print(f"[INFO] {remaining} faults to run across {len(pico_ports)} Pico(s)")

    if remaining == 0:
        print("[INFO] All faults already completed!")
        return

    est_time = remaining * 28 / len(pico_ports)
    print(f"[INFO] Estimated time: {est_time / 3600:.1f} hours ({est_time / 60:.0f} minutes)")

    # Open CSV
    csv_path = os.path.join(RESULTS_DIR, f"sweep_results_{CAMPAIGN_ID}.csv")
    csv_file = open(csv_path, mode="a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow([
            "Campaign_ID", "Tensor_Idx", "Category", "Dtype",
            "Fault_Num", "Weight_Idx", "Bit_Flipped",
            "Before_Value", "After_Value", "Accuracy", "Stop14_Accuracy"
        ])
        csv_file.flush()

    # Launch workers
    campaign_start = time.time()
    threads = []
    for idx, port in enumerate(pico_ports):
        t = threading.Thread(
            target=pico_worker,
            args=(idx + 1, port, fault_queue, completed_set, completed_lock,
                  csv_writer, csv_file),
            daemon=True,
        )
        threads.append(t)
        t.start()
        time.sleep(2)

    print(f"\n[INFO] {len(threads)} workers started. Ctrl+C to stop.\n")

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            with completed_lock:
                done = len(completed_set)
            if done >= total_faults:
                break

            # Auto-retry if all workers died
            if not any(t.is_alive() for t in threads) and not fault_queue.empty():
                print("\n[WARN] All workers exited. Retrying in 5s...")
                time.sleep(5)
                pico_ports = find_all_pico_ports()
                if pico_ports:
                    threads = []
                    for idx, port in enumerate(pico_ports):
                        t = threading.Thread(
                            target=pico_worker,
                            args=(idx + 1, port, fault_queue, completed_set, completed_lock,
                                  csv_writer, csv_file),
                            daemon=True,
                        )
                        threads.append(t)
                        t.start()
                        time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted. {len(completed_set)}/{total_faults} done. Resume available.")

    csv_file.close()
    elapsed = time.time() - campaign_start

    print(f"\n{'=' * 50}")
    print(f"SWEEP {'COMPLETE' if len(completed_set) >= total_faults else 'PARTIAL'}")
    print(f"{'=' * 50}")
    print(f"Faults completed  : {len(completed_set)}/{total_faults}")
    print(f"Total time        : {elapsed / 60:.1f} min ({elapsed / 3600:.1f} hrs)")
    print(f"Results CSV       : {csv_path}")

    if len(completed_set) >= total_faults and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    # Print quick vulnerability summary
    print(f"\n{'=' * 50}")
    print("VULNERABILITY SUMMARY (from CSV)")
    print(f"{'=' * 50}")
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        from collections import defaultdict
        tensor_stats = defaultdict(lambda: {"total": 0, "degraded": 0, "stop14_fail": 0})

        for row in rows:
            key = f"T{row['Tensor_Idx']} ({row['Category']})"
            tensor_stats[key]["total"] += 1

            acc_str = row.get("Accuracy", "").replace("%", "").strip()
            try:
                acc = float(acc_str)
                if acc < 90.8:
                    tensor_stats[key]["degraded"] += 1
            except ValueError:
                pass

            s14 = row.get("Stop14_Accuracy", "").replace("%", "").strip()
            try:
                s14_val = float(s14)
                if s14_val < 100.0:
                    tensor_stats[key]["stop14_fail"] += 1
            except ValueError:
                pass

        print(f"\n{'Tensor':<30} {'Faults':>7} {'Acc<90.8%':>10} {'Stop14<100%':>12}")
        print("-" * 62)
        for key in sorted(tensor_stats.keys(), key=lambda k: tensor_stats[k]["stop14_fail"], reverse=True):
            s = tensor_stats[key]
            deg_pct = s["degraded"] / s["total"] * 100 if s["total"] > 0 else 0
            s14_pct = s["stop14_fail"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"{key:<30} {s['total']:>7} {s['degraded']:>6} ({deg_pct:>4.0f}%) {s['stop14_fail']:>7} ({s14_pct:>4.0f}%)")

    except Exception as e:
        print(f"Could not generate summary: {e}")


if __name__ == "__main__":
    main()
