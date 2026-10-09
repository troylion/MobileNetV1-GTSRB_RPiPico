"""
Parallel SWIFI Fault Injection Campaign
========================================
Runs fault injection across multiple Raspberry Pi Picos simultaneously.
Uses a worker-pool model: each Pico pulls the next available fault from a
shared queue. If fewer Picos are available than expected, the available
ones keep working through the full queue. If a Pico disconnects, its
current fault is requeued and other Picos pick it up.

Usage:
    python run_campaign_parallel.py

Environment variables:
    NUM_FAULTS          Total faults to run (default: 1000)
    TENSOR_TO_ATTACK    Tensor index to target (default: 5)
    EXPERIMENT_MODE     Fault selection mode (default: 4)
    SKIP_FLASH          Skip compile+flash, "1" or "0" (default: "1")
    RESUME_CAMPAIGN     Resume from checkpoint, "1" or "0" (default: "1")
    PICO_PORTS          Comma-separated COM ports, e.g. "COM10,COM11,COM12"
                        If not set, auto-detects all connected Picos.
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

# Reuse fault_injector for model loading and RNG
import fault_injector

# ──────────────────────── CONFIGURATION ────────────────────────

NUM_FAULTS = int(os.environ.get("NUM_FAULTS", "1000"))
TENSOR_TO_ATTACK = int(os.environ.get("TENSOR_TO_ATTACK", "5"))
BAUD_RATE = 115200
SERIAL_TIMEOUT_SEC = 120
EXPERIMENT_MODE = int(os.environ.get("EXPERIMENT_MODE", "4"))
SKIP_FLASH = os.environ.get("SKIP_FLASH", "1") == "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICO_TFLMICRO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BUILD_DIR_NAME = "build"
BUILD_DIR = os.path.join(PICO_TFLMICRO_ROOT, BUILD_DIR_NAME)
UF2_PATH = os.path.join(BUILD_DIR, "examples", "mobilenet_gtsrb", "mobilenet_gtsrb.uf2")

USERHOME = os.path.expanduser("~")
NINJA_PATH = os.path.join(USERHOME, ".pico-sdk", "ninja", "v1.12.1", "ninja.exe")
PICOTOOL_PATH = os.path.join(USERHOME, ".pico-sdk", "picotool", "2.2.0-a4", "picotool", "picotool.exe")

CAMPAIGN_ID = time.strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "campaign_results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "campaign_parallel_checkpoint.json")
RESUME_CAMPAIGN = os.environ.get("RESUME_CAMPAIGN", "1") == "1"

# Resume from a specific CSV file (reads completed iterations from it)
# Example: RESUME_FROM=summary_results_20260427_110654.csv
RESUME_FROM = os.environ.get("RESUME_FROM", "")

# ──────────────────────── REGEX PATTERNS ────────────────────────

per_image_pattern = re.compile(
    r"^\[\s*(\d+)\s*/\s*(\d+)\]\s+(\S+)\s+GT:\s*(-?\d+)\s+Pred:\s*(-?\d+)\s+Conf:\s*(-?\d+)\s+\(\s*(\d+)%\)\s+(OK|WRONG)\s+\((\d+)\s+ms\)$"
)
stop14_pattern = re.compile(r"^Class\s+14:\s+\d+\s*/\s*\d+\s+\(([^)]+)\)")
stop14_new_pattern = re.compile(r"^STOP14:\s+\d+/\d+\s+\(([^)]+)\)")

# ──────────────────────── THREAD-SAFE CSV WRITER ────────────────────────

csv_lock = threading.Lock()
print_lock = threading.Lock()


def log(pico_id, msg):
    """Thread-safe print with Pico identifier."""
    with print_lock:
        print(f"[Pico {pico_id}] {msg}")


def write_summary_row(writer, file_handle, row):
    """Thread-safe CSV write."""
    with csv_lock:
        writer.writerow(row)
        file_handle.flush()


def write_per_image_row(writer, file_handle, row):
    """Thread-safe CSV write."""
    with csv_lock:
        writer.writerow(row)
        file_handle.flush()


# ──────────────────────── PICO DETECTION ────────────────────────

def find_all_pico_ports():
    """Find all connected Pico serial ports."""
    manual = os.environ.get("PICO_PORTS", "")
    if manual:
        return [p.strip() for p in manual.split(",") if p.strip()]

    ports = []
    for port in list_ports.comports():
        desc = (port.description or "").lower()
        hwid = (port.hwid or "").lower()
        # Pico shows up as "USB Serial Device" or with "2e8a" vendor ID
        if any(tok in desc for tok in ["pico", "usb serial", "cdc"]) or "2e8a" in hwid:
            ports.append(port.device)
    return sorted(ports)


# ──────────────────────── FAULT PARAMETER PRE-GENERATION ────────────────────────

def pregenerate_all_faults(num_faults, raw_bytes, experiment_mode, dtype):
    """Pre-generate all fault parameters deterministically.

    Returns a list of (iteration, rand_weight, actual_bit, byte_offset,
    bit_in_byte, before_value, after_value) tuples.
    """
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

    # Reset RNG to the campaign seed
    np.random.seed(fault_injector.RANDOM_SEED)

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

        # Compute before/after values
        weight_start = (byte_offset // itemsize) * itemsize
        orig_bytes = bytearray(raw_bytes[weight_start:weight_start + itemsize])
        before_value = np.frombuffer(bytes(orig_bytes), dtype=dtype)[0]

        faulted_bytes = bytearray(orig_bytes)
        byte_within = byte_offset - weight_start
        faulted_bytes[byte_within] ^= (1 << bit_in_byte)
        after_value = np.frombuffer(bytes(faulted_bytes), dtype=dtype)[0]

        before_val = float(before_value) if itemsize > 1 else int(before_value)
        after_val = float(after_value) if itemsize > 1 else int(after_value)

        faults.append((i, rand_weight, actual_bit, byte_offset, bit_in_byte, before_val, after_val))

    return faults


# ──────────────────────── CHECKPOINT ────────────────────────

def load_completed_iterations():
    """Load set of already-completed iteration indices and campaign ID from checkpoint."""
    if not RESUME_CAMPAIGN or not os.path.exists(CHECKPOINT_PATH):
        return set(), None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed", [])), data.get("campaign_id")
    except Exception:
        return set(), None


def save_completed_iterations(completed_set):
    """Save the set of completed iterations."""
    data = {
        "campaign_id": CAMPAIGN_ID,
        "num_faults": NUM_FAULTS,
        "tensor": TENSOR_TO_ATTACK,
        "completed": sorted(completed_set),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with csv_lock:  # reuse lock for file writes
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ──────────────────────── PICO WORKER ────────────────────────

def sync_pico(ser, pico_id):
    """Sync with a Pico by sending a junk command and waiting for READY_FOR_CMD."""
    ser.reset_input_buffer()
    time.sleep(0.3)
    ser.write(b"SYNC\n")

    deadline = time.time() + 15
    while time.time() < deadline:
        line_bytes = ser.readline()
        if not line_bytes:
            continue
        try:
            line = line_bytes.decode("utf-8", errors="replace").strip()
        except UnicodeDecodeError:
            continue
        if "READY_FOR_CMD" in line:
            return True
    return False


def pico_worker(pico_id, port, fault_queue, completed_set, completed_lock,
                faults_list, summary_writer, summary_file,
                per_image_writer, per_image_file):
    """Worker thread: connects to one Pico and processes faults from the queue."""
    try:
        # Reboot the Pico to get a clean CDC state
        log(pico_id, f"Rebooting {port}...")
        subprocess.run([PICOTOOL_PATH, "reboot", "-f"], capture_output=True)
        time.sleep(4)

        # Open serial
        log(pico_id, f"Connecting to {port}...")
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT_SEC)
        except serial.SerialException as e:
            log(pico_id, f"ERROR: Could not open {port}: {e}")
            return

        # Sync
        log(pico_id, "Syncing...")
        if not sync_pico(ser, pico_id):
            log(pico_id, "ERROR: Could not sync with Pico")
            ser.close()
            return

        # Query tensor size
        log(pico_id, f"Querying tensor {TENSOR_TO_ATTACK}...")
        ser.write(f"QUERY:{TENSOR_TO_ATTACK}\n".encode())
        query_deadline = time.time() + 10
        while time.time() < query_deadline:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line.startswith("SIZE:"):
                parts = line.split(":")[1].split(",")
                tensor_size = int(parts[1])
                log(pico_id, f"Tensor {TENSOR_TO_ATTACK} size: {tensor_size:,} bytes")
                break
            if "READY_FOR_CMD" in line:
                break

        # Wait for READY_FOR_CMD after query
        ready_deadline = time.time() + 10
        while time.time() < ready_deadline:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if "READY_FOR_CMD" in line:
                break

        # Set quiet mode
        log(pico_id, "Setting QUIET mode...")
        ser.write(b"MODE:QUIET\n")
        mode_deadline = time.time() + 5
        while time.time() < mode_deadline:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if "MODE:QUIET" in line:
                break

        log(pico_id, "Ready! Pulling faults from queue...")

        # Main loop: pull faults from queue
        faults_done = 0
        while True:
            try:
                fault_idx = fault_queue.get(timeout=2)
            except queue.Empty:
                # Check if all work is done
                with completed_lock:
                    if len(completed_set) >= NUM_FAULTS:
                        log(pico_id, "All faults completed. Exiting.")
                        break
                continue

            iteration, rand_weight, actual_bit, byte_offset, bit_in_byte, before_val, after_val = faults_list[fault_idx]
            fault_num = iteration + 1  # 1-indexed for display

            with completed_lock:
                total_done = len(completed_set)
            log(pico_id, f"FAULT {fault_num}/{NUM_FAULTS} (total done: {total_done}) "
                         f"weight[{rand_weight}] bit {actual_bit} "
                         f"({before_val} -> {after_val})")

            try:
                # Wait for READY_FOR_CMD
                ready_deadline = time.time() + 30
                ready = False
                while time.time() < ready_deadline:
                    line_bytes = ser.readline()
                    if not line_bytes:
                        continue
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if "READY_FOR_CMD" in line:
                        ready = True
                        break

                if not ready:
                    log(pico_id, f"WARN: Timeout waiting for READY_FOR_CMD. Requeueing fault {fault_num}.")
                    fault_queue.put(fault_idx)
                    break  # exit this worker, another Pico can pick it up

                # Send injection
                cmd = f"INJECT:{TENSOR_TO_ATTACK},{byte_offset},{bit_in_byte}\n"
                fault_start = time.time()
                ser.write(cmd.encode())

                # Read results
                final_accuracy = "ERROR"
                stop14_accuracy = "N/A"

                while True:
                    line_bytes = ser.readline()
                    if not line_bytes:
                        continue
                    try:
                        line = line_bytes.decode("utf-8", errors="replace").strip()
                    except UnicodeDecodeError:
                        continue

                    # Per-image results (verbose mode only, but parse if present)
                    match = per_image_pattern.match(line)
                    if match:
                        write_per_image_row(per_image_writer, per_image_file, [
                            CAMPAIGN_ID, fault_num,
                            int(match.group(1)), int(match.group(2)),
                            match.group(3), int(match.group(4)), int(match.group(5)),
                            int(match.group(6)), int(match.group(7)),
                            match.group(8), int(match.group(9)),
                        ])

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

                # Record result
                write_summary_row(summary_writer, summary_file, [
                    CAMPAIGN_ID, fault_num, TENSOR_TO_ATTACK,
                    rand_weight, actual_bit, before_val, after_val,
                    final_accuracy, stop14_accuracy,
                ])

                with completed_lock:
                    completed_set.add(iteration)
                    save_completed_iterations(completed_set)

                faults_done += 1
                log(pico_id, f"  -> Acc={final_accuracy}  Stop14={stop14_accuracy}  ({elapsed:.1f}s)  "
                             f"[{len(completed_set)}/{NUM_FAULTS} total]")

            except serial.SerialException as e:
                log(pico_id, f"ERROR: Serial error: {e}. Requeueing fault {fault_num}.")
                fault_queue.put(fault_idx)
                break
            except Exception as e:
                log(pico_id, f"ERROR: Unexpected: {e}. Requeueing fault {fault_num}.")
                fault_queue.put(fault_idx)
                break

        # Cleanup
        try:
            ser.close()
        except Exception:
            pass
        log(pico_id, f"Worker finished. Completed {faults_done} faults on this Pico.")

    except Exception as e:
        log(pico_id, f"FATAL ERROR: {e}")
        traceback.print_exc()


# ──────────────────────── MAIN ────────────────────────

def main():
    global CAMPAIGN_ID

    print("\n========== PARALLEL CAMPAIGN CONFIG ==========")

    # Detect Picos
    pico_ports = find_all_pico_ports()
    if not pico_ports:
        print("[ERROR] No Pico serial ports detected!")
        print("[TIP]   Set PICO_PORTS=COM10,COM11,COM12 to specify manually.")
        return

    print(f"Picos detected    : {len(pico_ports)} -> {', '.join(pico_ports)}")
    print(f"Num faults        : {NUM_FAULTS}")
    print(f"Tensor to attack  : {TENSOR_TO_ATTACK}")
    print(f"Experiment mode   : {EXPERIMENT_MODE}")
    print(f"Skip flash        : {SKIP_FLASH}")
    print(f"Resume enabled    : {RESUME_CAMPAIGN}")
    print(f"Resume from CSV   : {RESUME_FROM if RESUME_FROM else '(none)'}")
    print(f"Results folder    : {RESULTS_DIR}")
    print(f"Campaign ID       : {CAMPAIGN_ID}")
    print("=" * 48 + "\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model once to get tensor data
    print("[INFO] Loading model and pre-generating fault parameters...")
    buf, model = fault_injector.load_model(fault_injector.MODEL_PATH)
    tensor_dtype = fault_injector.get_tensor_dtype(model, TENSOR_TO_ATTACK)
    print(f"[INFO] Tensor {TENSOR_TO_ATTACK} dtype: {tensor_dtype.__name__}")
    _, raw_bytes = fault_injector.get_tensor_data_as_numpy(model, buf, TENSOR_TO_ATTACK, tensor_dtype)

    faults_list = pregenerate_all_faults(NUM_FAULTS, raw_bytes, EXPERIMENT_MODE, tensor_dtype)
    print(f"[INFO] Pre-generated {len(faults_list)} fault parameter sets (seed={fault_injector.RANDOM_SEED})")

    # Load checkpoint to skip already-completed faults
    completed_set = set()
    completed_lock = threading.Lock()

    # Method 1: Resume from a specific CSV file
    if RESUME_FROM:
        csv_path = RESUME_FROM
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(RESULTS_DIR, csv_path)
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            iteration = int(row[1])  # Iteration column (1-indexed in CSV)
                            completed_set.add(iteration - 1)  # Convert to 0-indexed
                        except ValueError:
                            continue
            # Extract campaign ID from filename (e.g., summary_results_20260427_110654.csv)
            basename = os.path.basename(csv_path)
            parts = basename.replace("summary_results_", "").replace(".csv", "")
            if parts:
                CAMPAIGN_ID = parts
            print(f"[INFO] Resuming campaign {CAMPAIGN_ID} from CSV: {len(completed_set)} faults already completed")
        else:
            print(f"[WARN] RESUME_FROM file not found: {csv_path}. Starting fresh.")

    # Method 2: Resume from checkpoint file
    elif RESUME_CAMPAIGN:
        loaded_set, prev_campaign_id = load_completed_iterations()
        if loaded_set and prev_campaign_id:
            completed_set = loaded_set
            CAMPAIGN_ID = prev_campaign_id
            print(f"[INFO] Resuming campaign {CAMPAIGN_ID} from checkpoint: {len(completed_set)} faults already completed")

    # Build the work queue (skip already-completed faults)
    fault_queue = queue.Queue()
    for i in range(NUM_FAULTS):
        if i not in completed_set:
            fault_queue.put(i)

    remaining = fault_queue.qsize()
    print(f"[INFO] {remaining} faults to run across {len(pico_ports)} Pico(s)")

    if remaining == 0:
        print("[INFO] All faults already completed!")
        return

    est_time = remaining * 28 / len(pico_ports)
    print(f"[INFO] Estimated time: {est_time / 3600:.1f} hours "
          f"({est_time / 60:.0f} minutes) at ~28s/fault")

    # Build firmware if needed
    if not SKIP_FLASH:
        print("\n[INFO] Building firmware...")
        result = subprocess.run([NINJA_PATH, "-C", BUILD_DIR, "examples/mobilenet_gtsrb/all"])
        if result.returncode != 0:
            print("[ERROR] Build failed!")
            return
        if not os.path.exists(UF2_PATH):
            print(f"[ERROR] UF2 not found: {UF2_PATH}")
            return
        print("[INFO] Build successful. NOTE: You must flash each Pico manually (BOOTSEL).")

    # Open CSV files
    summary_path = os.path.join(RESULTS_DIR, f"summary_results_{CAMPAIGN_ID}.csv")
    per_image_path = os.path.join(RESULTS_DIR, f"per_image_results_{CAMPAIGN_ID}.csv")

    summary_file = open(summary_path, mode="a", newline="", encoding="utf-8")
    per_image_file = open(per_image_path, mode="a", newline="", encoding="utf-8")
    summary_writer = csv.writer(summary_file)
    per_image_writer = csv.writer(per_image_file)

    # Write headers if files are new
    if os.path.getsize(summary_path) == 0:
        summary_writer.writerow([
            "Campaign_ID", "Iteration", "Tensor", "Weight_Idx", "Bit_Flipped",
            "Before_Value", "After_Value", "Accuracy", "Stop14_Accuracy"
        ])
        summary_file.flush()

    if os.path.getsize(per_image_path) == 0:
        per_image_writer.writerow([
            "Campaign_ID", "Fault_Iteration", "Image_Index", "Image_Total",
            "Filename", "GT_Class", "Pred_Class", "Confidence_Raw",
            "Confidence_Percent", "Status", "Inference_Time_ms"
        ])
        per_image_file.flush()

    # Launch worker threads
    campaign_start = time.time()
    threads = []
    for idx, port in enumerate(pico_ports):
        t = threading.Thread(
            target=pico_worker,
            args=(idx + 1, port, fault_queue, completed_set, completed_lock,
                  faults_list, summary_writer, summary_file,
                  per_image_writer, per_image_file),
            daemon=True,
        )
        threads.append(t)
        t.start()
        # Stagger startup slightly to avoid USB contention
        time.sleep(2)

    print(f"\n[INFO] {len(threads)} worker thread(s) started. Press Ctrl+C to stop.\n")

    # Wait for all workers to finish
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)

            # Check if queue is empty and all done
            with completed_lock:
                done = len(completed_set)
            if done >= NUM_FAULTS:
                break

            # If all threads died but work remains, try to respawn on available ports
            if not any(t.is_alive() for t in threads) and not fault_queue.empty():
                print("\n[WARN] All workers exited but faults remain. Retrying in 5s...")
                time.sleep(5)
                pico_ports = find_all_pico_ports()
                if pico_ports:
                    threads = []
                    for idx, port in enumerate(pico_ports):
                        t = threading.Thread(
                            target=pico_worker,
                            args=(idx + 1, port, fault_queue, completed_set, completed_lock,
                                  faults_list, summary_writer, summary_file,
                                  per_image_writer, per_image_file),
                            daemon=True,
                        )
                        threads.append(t)
                        t.start()
                        time.sleep(2)

    except KeyboardInterrupt:
        print("\n[INFO] Campaign interrupted by user.")
        print(f"[INFO] {len(completed_set)}/{NUM_FAULTS} faults completed. Resume available next run.")

    # Cleanup
    summary_file.close()
    per_image_file.close()

    elapsed = time.time() - campaign_start
    with completed_lock:
        done = len(completed_set)

    print(f"\n{'=' * 48}")
    print(f"CAMPAIGN {'COMPLETE' if done >= NUM_FAULTS else 'PARTIAL'}")
    print(f"{'=' * 48}")
    print(f"Faults completed  : {done}/{NUM_FAULTS}")
    print(f"Total time        : {elapsed / 60:.1f} minutes ({elapsed / 3600:.1f} hours)")
    if done > 0:
        print(f"Avg per fault     : {elapsed / done:.1f}s")
    print(f"Summary CSV       : {summary_path}")
    print(f"Per-image CSV     : {per_image_path}")

    if done >= NUM_FAULTS and os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("[INFO] All iterations completed. Checkpoint cleared.")


if __name__ == "__main__":
    main()
